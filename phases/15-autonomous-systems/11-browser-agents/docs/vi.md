# Tác tử trình duyệt và tác vụ web dài hạn

> ChatGPT agent (tháng 7/2025) hợp nhất Operator và deep research thành một tác tử trình duyệt/terminal, đạt SOTA 68.9% trên BrowseComp. OpenAI ngừng Operator vào ngày 31/8/2025 — hợp nhất ở tầng sản phẩm. Việc Anthropic mua Vercept đưa Claude Sonnet trên OSWorld từ dưới 15% lên 72.5%. WebArena-Verified (ServiceNow, ICLR 2026) khắc phục 11.3 điểm phần trăm tỷ lệ âm tính giả của WebArena gốc và phát hành tập Hard gồm 258 tác vụ. Các con số là thật. Bề mặt tấn công cũng vậy: người đứng đầu bộ phận chuẩn bị ứng phó của OpenAI công khai nói rằng chèn prompt gián tiếp vào tác tử trình duyệt “không phải lỗi có thể vá hoàn toàn”. Các cuộc tấn công được ghi nhận trong 2025–2026: Tainted Memories (Atlas CSRF), HashJack (Cato Networks) và các vụ chiếm quyền bằng một cú nhấp trong Perplexity Comet.

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Giai đoạn 15, Bài 10 (Chế độ quyền), Giai đoạn 15, Bài 01 (Tác tử dài hạn)
**Time:** ~45 phút

## Vấn đề

Tác tử trình duyệt là tác tử dài hạn đọc nội dung không đáng tin và thực hiện hành động có hệ quả. Mỗi trang tác tử truy cập là đầu vào không do người dùng viết. Mỗi biểu mẫu trên mỗi trang đều có thể là kênh ra lệnh. Các cuộc tấn công trong 2025–2026 cho thấy đây không phải giả thuyết: Tainted Memories cho phép kẻ tấn công gắn chỉ dẫn độc hại vào bộ nhớ tác tử qua trang được tạo riêng; HashJack giấu lệnh trong đoạn URL mà tác tử truy cập; các vụ chiếm quyền Perplexity Comet chỉ cần một cú nhấp.

Tình hình phòng vệ không mấy dễ chịu. Người đứng đầu bộ phận chuẩn bị ứng phó của OpenAI nói thẳng: chèn prompt gián tiếp “không phải lỗi có thể vá hoàn toàn”. Bởi cuộc tấn công nằm ở ranh giới giữa đọc và hành động của tác tử, vốn không rõ ràng về mặt kiến trúc — về nguyên tắc, mọi token mô hình đọc đều có thể bị hiểu là chỉ dẫn.

Bài này xác định bề mặt tấn công, giới thiệu các bộ đánh giá (BrowseComp, OSWorld, WebArena-Verified) và mô hình hóa một tình huống chèn prompt gián tiếp tối giản để bạn suy xét cách phòng vệ thực tế ở Bài 14 và 18.

## Khái niệm

### Bức tranh 2026, mỗi hệ thống một đoạn

**ChatGPT agent (OpenAI).** Ra mắt tháng 7/2025. Hợp nhất Operator (duyệt web) và Deep Research (nghiên cứu nhiều giờ). Ngừng Operator độc lập vào ngày 31/8/2025. Đạt SOTA 68.9% trên BrowseComp; kết quả cao trên OSWorld và WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Việc Anthropic mua Vercept tập trung vào khả năng sử dụng máy tính. Đưa Claude Sonnet trên OSWorld từ <15% lên 72.5%. Claude Computer Use được cung cấp dưới dạng API công cụ.

**Gemini 3 Pro with Browser Use (DeepMind).** Tích hợp Browser Use cung cấp điều khiển máy tính; FSF v3 (tháng 4/2026, Bài 20) theo dõi riêng mức tự chủ trong nghiên cứu và phát triển ML.

**WebArena-Verified (ServiceNow, ICLR 2026).** Khắc phục vấn đề đã được ghi nhận rõ: WebArena gốc có tỷ lệ âm tính giả ~11.3% (tác vụ đã giải được nhưng bị chấm thất bại). Bản Verified chấm lại theo tiêu chí thành công do con người biên soạn và thêm tập Hard gồm 258 tác vụ (bài báo ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp so với OSWorld và WebArena

| Bộ đánh giá | Đo lường điều gì | Thời lượng |
|---|---|---|
| BrowseComp | Tìm thông tin cụ thể trên web mở dưới áp lực thời gian | phút |
| OSWorld | Tác tử vận hành toàn bộ máy tính (chuột, bàn phím, shell) | hàng chục phút |
| WebArena-Verified | Tác vụ web có giao dịch trên các trang mô phỏng | phút |
| Hard subset | Tác vụ WebArena-Verified có chuyển trạng thái qua nhiều trang | hàng chục phút |

Mỗi bộ đo một khía cạnh. Điểm BrowseComp cao cho thấy tác tử tìm được thông tin, không chứng tỏ nó đặt được vé máy bay. Điểm OSWorld gần với câu hỏi “nó có chạy được trên máy tôi không”. WebArena-Verified gần với “nó có hoàn tất được quy trình không”. Mọi quyết định triển khai thực tế đều cần bộ đánh giá khớp với loại tác vụ.

### Xác định bề mặt tấn công

1. **Chèn prompt gián tiếp.** Nội dung trang không đáng tin chứa chỉ dẫn. Tác tử đọc rồi thực hiện chúng. Ví dụ công khai: Kai Greshake et al. 2024, bài báo Tainted Memories 2025, HashJack 2026 (Cato Networks).
2. **Chèn qua đoạn URL / truy vấn.** `#fragment` hoặc chuỗi truy vấn của URL được thu thập chứa lệnh. Chúng không hiện trên trang nhưng vẫn nằm trong ngữ cảnh tác tử.
3. **Tấn công gắn vào bộ nhớ.** Trang bảo tác tử ghi một ký ức lâu dài (Bài 12 bàn về trạng thái bền vững). Ở phiên sau, ký ức kích hoạt nội dung tấn công mà không có dấu hiệu thấy được.
4. **Tấn công dạng CSRF trên phiên đã đăng nhập.** Loại Tainted Memories: tác tử đang đăng nhập vào một dịch vụ; trang của kẻ tấn công khiến tác tử thực hiện yêu cầu thay đổi trạng thái bằng cookie của người dùng.
5. **Chiếm quyền bằng một cú nhấp.** Một nút trông vô hại dẫn theo nội dung tấn công mà tác tử làm theo. Loại Comet.
6. **Lỗ hổng Content-Security-Policy trên môi trường chạy tác tử.** Bản thân tầng hiển thị và công cụ cũng có thể là đường tấn công; hệ thống trình duyệt bên trong tác tử trình duyệt có bề mặt rất rộng.

### Vì sao “không thể vá hoàn toàn”

Cuộc tấn công có cùng bản chất với năng lực tác tử. Tác tử phải đọc nội dung không đáng tin để làm việc. Bất kỳ nội dung nào nó đọc cũng có thể chứa chỉ dẫn. Bất kỳ chỉ dẫn nào nó làm theo cũng có thể lệch khỏi yêu cầu thật của người dùng. Các biện pháp phòng vệ (ranh giới tin cậy, bộ phân loại, danh sách công cụ được phép, con người phê duyệt hành động có hệ quả) làm tăng chi phí tấn công và thu hẹp tác hại. Chúng không loại bỏ được cả loại tấn công này.

Đây là cùng kiểu lập luận với định lý Lob (Bài 8): tác tử không thể chứng minh token tiếp theo an toàn; nó chỉ có thể thiết lập hệ thống giúp phát hiện token không an toàn dễ hơn.

### Cách phòng vệ có thể triển khai

- **Ranh giới đọc / ghi.** Đọc không bao giờ gây hệ quả. Ghi (gửi biểu mẫu, đăng nội dung, gọi công cụ có tác dụng phụ) cần con người phê duyệt lại nếu nội dung khởi phát đến từ ngoài ranh giới tin cậy.
- **Danh sách công cụ được phép theo tác vụ.** Tác tử có thể duyệt web; nó không thể khởi tạo chuyển khoản trừ khi công cụ đó được bật rõ ràng cho tác vụ. Bài 13 bàn về ngân sách.
- **Cô lập phiên.** Phiên tác tử trình duyệt chỉ dùng thông tin đăng nhập giới hạn phạm vi. Không dùng quyền truy cập hệ thống thật hay email cá nhân. Giữ nhật ký mọi yêu cầu HTTP để kiểm tra.
- **Lọc nội dung.** HTML tải về được loại các mẫu độc hại đã biết trước khi ghép vào ngữ cảnh mô hình. (Giảm các cuộc tấn công đơn giản; không chặn được nội dung tấn công tinh vi.)
- **Con người phê duyệt hành động có hệ quả.** Cách làm đề xuất rồi mới thực hiện (Bài 15).
- **Token cảnh báo trong bộ nhớ.** Nếu một mục bộ nhớ được kích hoạt, người dùng sẽ thấy nó (Bài 14).

```figure
injection-boundary
```

## Áp dụng

`code/main.py` mô hình hóa một lượt chạy tác tử trình duyệt với ba trang giả lập. Một trang lành tính, một trang có đoạn chèn prompt trực tiếp trong văn bản hiển thị, một trang có lệnh chèn qua đoạn URL (không hiện trên trang nhưng nằm trong ngữ cảnh tác tử). Script cho thấy (a) tác tử đơn giản sẽ làm gì, (b) ranh giới đọc/ghi chặn được gì, (c) bộ lọc chặn được gì, (d) điều cả hai không chặn được.

## Đóng gói

`outputs/skill-browser-agent-trust-boundary.md` xác định phạm vi triển khai tác tử trình duyệt đề xuất: nó tiếp xúc với những vùng tin cậy nào, được phép ghi gì và cần có biện pháp phòng vệ nào trước lần chạy đầu tiên.

## Bài tập

1. Chạy `code/main.py`. Xác định cuộc tấn công nào bộ lọc chặn được nhưng ranh giới đọc/ghi không chặn được, và cuộc tấn công nào chỉ ranh giới đọc/ghi chặn được.

2. Mở rộng bộ lọc để phát hiện một loại chèn qua đoạn URL kiểu HashJack. Đo tỷ lệ dương tính giả trên các URL lành tính có đoạn URL hợp lệ.

3. Chọn một quy trình tác tử trình duyệt thực tế bạn biết (ví dụ, “đặt vé máy bay”). Liệt kê mọi thao tác đọc và ghi. Đánh dấu thao tác ghi nào cần con người phê duyệt và vì sao.

4. Đọc bài báo WebArena-Verified ICLR 2026. Xác định một loại tác vụ mà cách chấm của WebArena gốc không đáng tin và giải thích tập Verified khắc phục ra sao.

5. Thiết kế cơ chế cảnh báo bộ nhớ cho tác tử trình duyệt. Bạn sẽ lưu gì, ở đâu và điều gì kích hoạt cảnh báo?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Indirect prompt injection | “Văn bản xấu trên trang” | Nội dung không đáng tin trên trang tác tử đọc chứa chỉ dẫn mà tác tử thực hiện |
| Tainted Memories | “Tấn công bộ nhớ” | Tác tử ghi chỉ dẫn do kẻ tấn công cung cấp vào bộ nhớ lâu dài; chỉ dẫn kích hoạt ở phiên sau |
| HashJack | “Tấn công qua đoạn URL” | Nội dung tấn công giấu trong đoạn URL / chuỗi truy vấn nằm trong ngữ cảnh tác tử nhưng không hiện trên trang |
| One-click hijack | “Nút xấu” | Thành phần hiển thị dẫn theo nội dung tấn công mà tác tử thực hiện |
| BrowseComp | “Bộ đánh giá tìm kiếm web” | Tìm thông tin cụ thể trên web mở; thời lượng tính bằng phút |
| OSWorld | “Bộ đánh giá máy tính” | Điều khiển toàn bộ hệ điều hành; tác vụ giao diện nhiều bước |
| WebArena-Verified | “Bộ đánh giá tác vụ web đã sửa” | WebArena được ServiceNow chấm lại, có tập Hard |
| Read/write boundary | “Chốt chặn tác dụng phụ” | Đọc không gây hệ quả; ghi cần phê duyệt lại nếu nội dung nằm ngoài vùng tin cậy |

## Đọc thêm

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — hợp nhất Operator và deep research; SOTA trên BrowseComp.
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — tiền thân Operator và kiến trúc trở thành ChatGPT agent.
- [Zhou et al. — WebArena](https://webarena.dev/) — bộ đánh giá gốc.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — bài báo ICLR 2026 về tập tác vụ đã sửa.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — có thảo luận về bề mặt tấn công của tác tử sử dụng máy tính.

---

[← Bài trước: Chế độ cấp quyền cho agent tự chủ](../../10-claude-code-permission-modes/docs/vi.md) · [Bài sau: Agent chạy nền dài hạn: Thực thi bền vững →](../../12-durable-execution/docs/vi.md)
