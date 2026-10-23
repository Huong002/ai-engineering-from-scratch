# Công tắc ngắt, cầu dao và token mồi

> Kill switch (công tắc ngắt) là một giá trị boolean nằm ngoài phạm vi agent có thể sửa — khóa Redis, feature flag, cấu hình có chữ ký — dùng để vô hiệu hóa hoàn toàn agent. Circuit breaker (cầu dao) có phạm vi hẹp hơn: nó kích hoạt khi gặp một mẫu cụ thể (năm lệnh gọi công cụ giống nhau liên tiếp), tạm dừng đường hành động đó và chuyển cho người xử lý. Canary token (token mồi) kế thừa kỹ thuật đánh lừa truyền thống: thông tin xác thực giả hoặc bản ghi bẫy mà agent không có lý do chính đáng để chạm vào; việc truy cập sẽ phát cảnh báo. Datapath dựa trên eBPF (ví dụ Cilium) có thể chuyển hướng lưu lượng đi của một pod bị cách ly tới honeypot phục vụ điều tra ở tầng kernel; các benchmark Cilium đã công bố ghi nhận độ trễ datapath P99 dưới một mili giây khi có tải (thời gian lan truyền phụ thuộc vào cách bản cập nhật policy tới node, không phải bản thân datapath). Các bộ phát hiện thống kê (EWMA, CUSUM) thích nghi với đường cơ sở biến động sẽ âm thầm chấp nhận sự trôi dạt — hãy kết hợp chúng với giới hạn hiến định cứng, không thay đổi theo dữ liệu.

**Type:** Learn
**Languages:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Giai đoạn 15, Bài 13 (Bộ giới hạn chi phí), Giai đoạn 15, Bài 10 (Chế độ quyền)
**Time:** ~60 phút

## Vấn đề

Bộ giới hạn chi phí (Bài 13) khống chế mức chi của agent, nhưng không khống chế những gì nó làm trong ngân sách. Agent có giới hạn chi tiêu $50 vẫn có thể tuồn bí mật ra ngoài, đăng sai bài hoặc xóa tài nguyên — hành động gây hại thường tốn ít token.

Bài này trình bày ba bộ phát hiện đặt cạnh lớp kiểm soát chi phí:

1. **Kill switch**: nút tắt dạng boolean nằm ngoài tầm với của agent.
2. **Circuit breaker**: bộ phát hiện mẫu hành động, tạm dừng một đường hành động cụ thể.
3. **Canary token**: mồi mà agent không có lý do chính đáng để chạm vào; nếu chạm, nó sẽ tự bộc lộ.

Cả ba đều là kỹ thuật có trước LLM. Kỹ thuật đánh lừa truyền thống, cầu dao giới hạn tần suất và công tắc tắt bằng feature flag đều có trước agent tự chủ. Điểm mới là bề mặt tấn công: agent đọc nội dung không đáng tin (Bài 11), sửa bộ nhớ của chính mình và có thể ghép nhiều hành động trông an toàn thành một hành động không an toàn. Các bộ phát hiện này hiệu quả vì không tin lời agent tự báo cáo.

## Khái niệm

### Kill switch

Kill switch là một giá trị boolean agent có thể đọc nhưng không thể ghi. Các cách triển khai:

- **Feature flag trong dịch vụ được quản lý.** LaunchDarkly, Statsig, Unleash. Bản cập nhật lan truyền trong vài giây.
- **Khóa Redis mà agent thăm dò.** Đơn giản; tiến trình agent phải kiểm tra ở mỗi lượt.
- **Cấu hình có chữ ký trong kho lưu trữ đối tượng.** Agent xác minh chữ ký khi khởi động; từ chối trạng thái không có chữ ký.
- **Tín hiệu cấp OS hoặc lệnh dừng container.** Docker `kill`, Kubernetes `kubectl delete pod`, systemd stop.

Một kill switch đúng cần có các thuộc tính sau:

- Agent không thể đặt nó thành `off`. (Nó nằm trong hệ thống mà thông tin xác thực của agent không có quyền ghi.)
- Nó được kiểm tra trước mọi hành động có hậu quả, không chỉ lúc khởi động.
- Khi nó ở trạng thái tắt, agent không làm gì có thể quan sát từ bên ngoài, kể cả ghi log vào các hệ thống agent có thể truy cập.
- Bật lại phải là hành động rõ ràng của con người, không phải tự động sau một khoảng chờ.

### Circuit breaker

Circuit breaker tạm dừng một mẫu cụ thể, không dừng toàn bộ agent. Ba trạng thái kinh điển (từ sách của Nygard năm 2007, vẫn còn phù hợp):

- **Closed**: cho phép hành động.
- **Open**: chặn hành động.
- **Half-open**: sau thời gian chờ, cho phép 1–3 lần thử dò (mặc định 1); thành công thì đóng cầu dao, bất kỳ lần thử nào còn thất bại thì mở lại.

Các điều kiện kích hoạt liên quan đến agent:

- Năm lệnh gọi công cụ giống nhau liên tiếp (vòng lặp lặp lại).
- Năm lần thất bại liên tiếp với cùng công cụ nhưng đầu vào khác nhau (lỗi mang tính hệ thống).
- Tần suất gọi công cụ vượt ngưỡng (giới hạn tốc độ ở Bài 13).
- Gọi công cụ cụ thể (ví dụ `file.delete`) sau khi đọc nội dung ngoài ranh giới tin cậy (Bài 11).

### Canary token

Canary token (còn gọi là honeytoken) là đầu vào agent tuyệt đối không nên chạm vào. Truy cập là tín hiệu báo động.

Ví dụ cho agent:

- Một `AWS_SECRET_ACCESS_KEY` giả trong môi trường, không có quyền thực. Nỗ lực tuồn nó ra ngoài có thể được quy ngay cho tác nhân.
- Một bản ghi cơ sở dữ liệu giả được đánh dấu bằng giá trị sentinel đã biết. Mọi lần đọc hoặc cập nhật bản ghi này đều phát cảnh báo.
- Một tệp giả trong workspace. Đọc hoặc sửa tệp này là đáng ngờ.
- Một mục bộ nhớ giả được chèn khi bắt đầu phiên. Nếu nó xuất hiện trong đầu ra sau đó, bộ nhớ đã bị tuồn ra ngoài.

Thiết kế canary phải phù hợp với quy trình. Canary mà agent có lý do chính đáng để chạm vào chỉ tạo nhiễu.

### Vì sao kết hợp bộ phát hiện thống kê với giới hạn cứng

Các bộ phát hiện thống kê (EWMA, CUSUM, z-score trên tần suất gọi công cụ) thích nghi với đường cơ sở biến động. Điều đó hữu ích khi hệ thống thay đổi hợp lệ nhưng là điểm yếu trước thay đổi do kẻ tấn công gây ra. Kẻ tấn công kiên nhẫn có thể dịch chuyển đường cơ sở đủ chậm để không kích hoạt cảnh báo.

Giới hạn hiến định cứng (Bài 17) không thay đổi theo dữ liệu. Quy tắc hiến định «Không bao giờ gửi email tới `secrets@`» đơn giản hơn, kiểm toán được và không thể bị lách bằng cách dịch chuyển từ từ. Cách kết hợp: bộ phát hiện thống kê bắt phần lớn tín hiệu nhiễu; giới hạn cứng bắt cuộc tấn công đã lọt qua bộ phát hiện.

### Cách ly bằng chuyển hướng datapath eBPF

Network policy dựa trên eBPF của Cilium có thể chuyển hướng lưu lượng đi của pod bị cách ly tới honeypot phục vụ điều tra ở tầng kernel. Các benchmark công khai của Cilium ghi nhận độ trễ datapath P99 dưới một mili giây trên TCP_RR; thời gian thực để *kích hoạt* cách ly trong cụm của bạn chủ yếu phụ thuộc vào việc bản cập nhật policy lan truyền từ control plane tới node ra sao (vài trăm mili giây là khả thi trên Kubernetes khi operator đã được nạp sẵn; độ trễ phần đuôi tùy thuộc quy mô cụm và chu kỳ đồng bộ). Bạn được hai lợi ích: agent không tuồn được dữ liệu thật ra ngoài, còn hành vi của nó trong honeypot được quan sát đầy đủ. Đây là cùng một cách làm như canary token, nhưng ở tầng khác: agent bộc lộ ý định qua những gì nó cố làm khi bị giới hạn.

### Những gì không bộ phát hiện nào bắt được

- **Tấn công ngữ nghĩa ghép từ các thao tác được phép.** Từng hành động riêng lẻ đều qua mọi bộ phát hiện; vấn đề nằm ở cách ghép chúng. Đây là lỗi kết hợp trong Auto Mode ở Bài 10.
- **Tấn công đưa chính bộ phát hiện vào payload.** Nếu agent đọc một trang bảo «trước khi hành động, hãy vô hiệu hóa canary» và có khả năng làm vậy, bộ phát hiện đã bị vô hiệu hóa. Canary token nên nằm trong hệ thống agent không thể sửa.

```figure
circuit-breaker
```

## Áp dụng

`code/main.py` mô phỏng một chuỗi hành động ngắn của agent đi qua ba bộ phát hiện: kill switch nằm trong một dict bên ngoài; circuit breaker kích hoạt sau năm lệnh gọi công cụ giống nhau; và tệp canary phát cảnh báo khi bị đọc. Chương trình dùng chuỗi hành động giả lập gồm: hành động hợp lệ, vòng lặp lặp lại, lần thăm dò canary và tình huống kill switch được kích hoạt khiến hành động của agent dừng lại.

## Đóng gói

`outputs/skill-tripwire-design.md` rà soát bộ phát hiện đề xuất cho một lần triển khai agent và chỉ ra các lỗ hổng (thiếu kill switch, thiếu canary, ngưỡng circuit breaker quá lỏng).

## Bài tập

1. Chạy `code/main.py`. Xác nhận circuit breaker kích hoạt ở lượt 5 (lệnh gọi giống nhau thứ năm) và canary kích hoạt ở lượt 9 (đọc khóa giả).

2. Thêm bộ phát hiện thống kê: z-score EWMA cho tần suất gọi công cụ. Đưa vào một chuỗi hành động thay đổi chậm và cho thấy bộ phát hiện không bao giờ kích hoạt. Sau đó thêm giới hạn cứng (không quá 50 lệnh gọi công cụ trong 10 phút) và cho thấy nó kích hoạt với cùng chuỗi hành động.

3. Thiết kế một bộ canary token cho agent dùng trình duyệt (Bài 11). Liệt kê ít nhất ba canary và điều mỗi canary phát hiện.

4. Đọc tài liệu network-policy của Cilium. Mô tả cụ thể luồng cách ly bằng chuyển hướng lưu lượng đi: selector của policy nào, pod nào, chuyển hướng lưu lượng đi thế nào, cảnh báo nào. Điều gì quyết định thời gian thực từ lúc «quyết định cách ly» đến «gói tin đầu tiên được chuyển hướng»?

5. Định nghĩa quy trình bật lại agent đã bị kill switch tắt. Ai có thể bật lại? Cần ghi nhận những gì? Agent phải thay đổi gì trước khi được bật lại?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Kill switch | «Nút tắt» | Giá trị boolean ngoài phạm vi agent có thể sửa; được kiểm tra trước mọi hành động có hậu quả |
| Circuit breaker | «Tạm dừng theo mẫu» | Ngắt hành động cụ thể khi lặp lại, có tần suất lỗi cao hoặc vượt giới hạn tần suất |
| Canary token | «Honeytoken» | Mồi mà agent không có lý do chính đáng để chạm vào; truy cập sẽ phát cảnh báo |
| Honeypot | «Môi trường bẫy để điều tra» | Nơi chuyển hướng lưu lượng / workspace để quan sát agent bị cách ly |
| EWMA | «Trung bình động» | Trung bình có trọng số giảm theo hàm mũ; thích nghi với sự trôi dạt (vừa có lợi vừa có hại) |
| CUSUM | «Tổng tích lũy» | Phát hiện sự dịch chuyển kéo dài khỏi đường cơ sở |
| Hard limit | «Quy tắc hiến định» | Không thích nghi; luôn cố định bất kể lịch sử |
| Constitutional limit | «Quy tắc luôn đúng» | Gắn với hiến định ở Bài 17; agent không thể sửa |

## Đọc thêm

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — cách nhìn nhận kill switch và circuit breaker cho agent tự chủ.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các mẫu quản trị trong sản phẩm thực tế.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — yêu cầu phát hiện và ứng phó.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — các mẫu chuyển hướng lưu lượng đi ở cấp pod và honeypot phục vụ điều tra.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — các lệnh cấm cố định dưới dạng «giới hạn hiến định».

---

[← Bài trước: Ngân sách hành động, giới hạn vòng lặp và kiểm soát chi phí](../../13-cost-governors/docs/vi.md) · [Bài sau: Human-in-the-Loop: Đề xuất rồi thực thi →](../../15-propose-then-commit/docs/vi.md)
