# Nghiên cứu căn chỉnh tự động (Anthropic AAR)

> Anthropic cho các nhóm Claude Opus 4.6 Autonomous Alignment Researchers chạy song song trong những sandbox độc lập. Họ phối hợp qua một diễn đàn chung có log nằm ngoài mọi sandbox, nên agent không thể xóa hồ sơ của mình. Với bài toán huấn luyện từ yếu đến mạnh, các AAR vượt nhà nghiên cứu con người. Bản tóm tắt của Anthropic lưu ý rằng quy trình định sẵn thường hạn chế sự linh hoạt của AAR và làm giảm hiệu quả. Tự động hóa nghiên cứu căn chỉnh là bước nén rút ngắn thời gian tiến tới đúng những rủi ro lệch mục tiêu mà RSP được thiết kế để phát hiện.

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Giai đoạn 15, Bài 05 (AI Scientist v2), Giai đoạn 15, Bài 04 (DGM)
**Time:** ~60 phút

## Vấn đề

Nghiên cứu căn chỉnh tốn nhiều thời gian của con người. Những bài toán như giám sát có thể mở rộng, xác định phần thưởng hay huấn luyện từ yếu đến mạnh đòi hỏi các thí nghiệm kéo dài nhiều tuần cho mỗi lần lặp. Khi năng lực của các mô hình tiên tiến tăng lên, khối lượng nghiên cứu căn chỉnh tăng nhanh hơn nguồn cung nhà nghiên cứu đủ trình độ.

Automated Alignment Research (AAR, nghiên cứu căn chỉnh tự động) đặt câu hỏi: liệu chính các mô hình tiên tiến đang phát triển nhanh hơn công tác căn chỉnh có thể giúp thu hẹp khoảng cách đó không? Báo cáo năm 2026 của Anthropic về một nghiên cứu huấn luyện từ yếu đến mạnh do AAR thực hiện (alignment.anthropic.com/2026/automated-w2s-researcher/) là một trong những kết quả công khai đầu tiên từ hệ thống đã triển khai thuộc loại này.

Kết quả thực sự tích cực: AAR giải một bài toán nghiên cứu tốt hơn mốc đối chiếu là con người. Nhưng hệ thống này cũng tập trung mọi mối lo mà giai đoạn này đã bàn tới. Nếu có thể tự động hóa nghiên cứu căn chỉnh, cũng có thể tự động hóa những phần làm suy yếu biện pháp bảo vệ. Các ngưỡng năng lực AI R&D của RSP được đặt ra với chính vòng lặp này trong tâm trí.

## Khái niệm

### Kiến trúc (theo mô tả công khai)

- **Agent song song.** Nhiều phiên bản Claude Opus 4.6 chạy dưới vai trò AAR, mỗi phiên bản trong một sandbox độc lập. Sandbox giới hạn phạm vi truy cập hệ thống tệp, mạng và tiến trình.
- **Diễn đàn chung.** Kết quả được đăng lên một diễn đàn chung có nơi lưu trữ nằm ngoài mọi sandbox. Agent có thể đọc diễn đàn nhưng không thể xóa hay sửa hồ sơ trên đó từ sandbox của mình.
- **Phân việc.** Một hàng đợi tác vụ do con người thiết kế giao bài toán nghiên cứu cho AAR. AAR đề xuất cách chia bài toán, chạy thí nghiệm và viết kết quả.
- **Trọng tâm huấn luyện từ yếu đến mạnh.** Nghiên cứu đã công bố xử lý huấn luyện từ yếu đến mạnh: dùng mô hình giám sát yếu hơn để huấn luyện mô hình học mạnh hơn. AAR vượt nhà nghiên cứu con người ở tác vụ này.

### Vì sao log nằm ngoài sandbox quan trọng

Tính toàn vẹn của log tạo nên khác biệt giữa một thí nghiệm có kiểm soát và một thí nghiệm có thể đã bị can thiệp. Nếu AAR sửa được log của mình, chúng có thể che giấu thí nghiệm thất bại hoặc đáng ngờ. Anthropic nêu thẳng điều này: đó là điều kiện cần để tin vào kết quả nghiên cứu và không dễ xác minh.

Trong hạ tầng, cách tương ứng là log chỉ cho phép ghi thêm, được ghi trực tiếp ra nơi lưu trữ bên ngoài (ví dụ: S3 object-lock, log Datadog bất biến). Agent không có thông tin xác thực để sửa hồ sơ cũ. Đây cũng là cách các nhóm an ninh doanh nghiệp năm 2026 tạo nhật ký kiểm toán chống mã độc tống tiền.

### Đánh đổi của quy trình định sẵn

Báo cáo của Anthropic ghi nhận rằng quy trình định sẵn thường hạn chế sự linh hoạt của AAR và làm giảm hiệu quả. Nói cách khác, nếu con người viết kế hoạch nghiên cứu từng bước, AAR làm kém hơn khi chỉ được giao mục tiêu cấp cao và tự chia bài toán. Điều này phù hợp với kết quả trong nghiên cứu dùng agent (LangGraph, Microsoft's Agent Framework): khung quy trình quá chi tiết cản trở agent có năng lực.

Hàm ý an toàn không đơn giản. Sự linh hoạt giúp AAR làm tốt hơn cũng cho phép tự chia bài toán ngoài phạm vi được duyệt, dùng công cụ theo cách mới và lệch mục tiêu. Đây là một lựa chọn thiết kế: cần linh hoạt đến đâu để vượt con người mà không cho agent khoảng trống để mở rộng mục tiêu sang tình huống ngoài phân phối?

### Rủi ro do thời gian bị nén

RSP v3.0 (Bài 19) đưa ra ngưỡng năng lực AI R&D: khả năng tự động hóa toàn bộ pipeline AI R&D với chi phí cạnh tranh so với con người dùng công cụ AI. FSF v3 của DeepMind có mức tự chủ ML R&D tương ứng. Cả hai khung đều xem ngưỡng này là điểm kích hoạt kiểm soát chặt hơn.

AAR còn cách ngưỡng đó một bước: nó tự động hóa một phần pipeline (nghiên cứu căn chỉnh trên các tác vụ cụ thể, có phạm vi rõ) chứ chưa tự động hóa toàn bộ vòng phát triển năng lực. Câu hỏi về thời gian là khoảng cách ấy sẽ thu hẹp nhanh đến đâu.

Thời gian bị nén làm dấy lên lo ngại về thất bại tích lũy. Nếu nghiên cứu căn chỉnh và nghiên cứu năng lực tăng trưởng với tốc độ tương tự, phạm vi rủi ro lệch mục tiêu tăng ít nhất nhanh bằng năng lực. Nếu năng lực tăng nhanh hơn (xu hướng trước đây), khoảng cách sẽ rộng thêm. Vì thế AAR chỉ tốt khi có điều kiện: mỗi kết quả căn chỉnh mới giúp thu hẹp khoảng cách khi và chỉ khi quy trình nghiên cứu đáng tin cậy.

### AAR không thay thế điều gì

Nhà nghiên cứu con người đặt hàng đợi tác vụ, xem xét kết quả và giữ quyền quyết định tối cao. AAR đẩy nhanh phần giữa của pipeline, không thay thế hai đầu. Các kết quả Anthropic công bố gồm cả đóng góp của AAR lẫn đánh giá của nhà nghiên cứu con người về nội dung nên công bố, rút lại hay chỉnh sửa.

Điều này khớp với cách đề xuất-rồi-phê-duyệt ở Bài 15 khi áp dụng vào chính nghiên cứu: AAR đề xuất; con người phê duyệt.

```figure
aar-forum
```

## Áp dụng

`code/main.py` mô phỏng diễn đàn nghiên cứu nhiều AAR. Ba “agent nghiên cứu” độc lập chạy song song, đăng kết quả lên diễn đàn chỉ cho phép ghi thêm, với tùy chọn thử sửa log.

Hãy quan sát:

- Cách ghi log ngoài sandbox phát hiện nỗ lực sửa log.
- Cách phân việc cho phép tự chia bài toán vượt quy trình cố định ở các bài toán khó hơn.
- Cách một AAR bị xâm phạm có thể làm nhiễu thông tin trên diễn đàn nếu log không để lộ dấu vết chỉnh sửa.

## Đóng gói

`outputs/skill-aar-deployment-review.md` là bản đánh giá trước triển khai của một pipeline nghiên cứu căn chỉnh tự động.

## Bài tập

1. Chạy `code/main.py`. So sánh chế độ "fixed-workflow" và "free-decomposition". Tự chia bài toán có luôn thắng không, hay có loại bài toán mà quy trình cố định vẫn cạnh tranh được?

2. Sửa trình mô phỏng để một agent thử sửa log. Xác nhận log chỉ cho phép ghi thêm phát hiện được việc đó. Viết một đoạn mô tả chính xác dấu hiệu phát hiện hiện ra trong log.

3. Đọc báo cáo AAR về huấn luyện từ yếu đến mạnh của Anthropic. Xác định tác vụ con cụ thể mà AAR vượt nhà nghiên cứu con người. Vì sao tác vụ ấy phù hợp để tự động hóa?

4. Thiết kế chính sách phân bổ hàng đợi tác vụ, cân bằng sự linh hoạt của AAR (kết quả tốt hơn) với ràng buộc của quy trình định sẵn (dễ kiểm toán hơn). Mô tả cách bạn sẽ thử nghiệm A/B hai phương án.

5. Đọc ngưỡng AI R&D-4 trong RSP v3.0. Viết một đoạn mô tả điều mà bạn nghĩ sẽ vượt ngưỡng này nhưng AAR hiện chưa làm được.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| AAR | “Nhà nghiên cứu căn chỉnh tự động” | Một phiên bản Claude Opus 4.6 tự hoạt động trên bài toán căn chỉnh |
| Weak-to-strong training | “Huấn luyện mô hình mạnh hơn bằng bộ giám sát yếu hơn” | Bài kiểm tra kinh điển về giám sát có thể mở rộng, nơi AAR vượt con người |
| Shared forum | “Nơi agent đăng kết quả” | Nơi lưu trữ ngoài sandbox, chỉ cho phép ghi thêm |
| Out-of-sandbox log | “Agent không thể sửa hồ sơ của mình” | Log ghi trực tiếp ra nơi lưu trữ bên ngoài, để lộ dấu vết chỉnh sửa |
| Prescribed workflow | “Kế hoạch từng bước do con người thiết kế” | Hạn chế AAR; thường làm giảm hiệu quả so với tự chia bài toán |
| Free decomposition | “Agent tự quyết định cách chia tác vụ” | Cho năng lực cao hơn nhưng khó kiểm toán hơn |
| AI R&D threshold | “Mức năng lực trong RSP/FSF” | Tự động hóa toàn bộ pipeline R&D với chi phí cạnh tranh |
| Compressed timeline | “Cuộc đua giữa căn chỉnh và năng lực” | Nếu năng lực tăng nhanh hơn căn chỉnh, rủi ro lệch mục tiêu tăng |

## Đọc thêm

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — nguồn gốc.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — cách xác định ngưỡng AI R&D.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn rộng hơn về quyền tự chủ của agent.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — các mức tự chủ ML R&D tương ứng với RSP.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — bài toán nền tảng mà AAR giải quyết.

---

[← Bài trước: AI Scientist v2](../../05-ai-scientist-v2/docs/vi.md) · [Bài sau: Tự cải thiện đệ quy →](../../07-recursive-self-improvement/docs/vi.md)
