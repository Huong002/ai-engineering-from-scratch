# Human-in-the-Loop: Đề xuất rồi thực thi

> Đồng thuận năm 2026 về HITL (con người tham gia vòng lặp) rất cụ thể. Đây không phải kiểu “agent hỏi, người dùng bấm Approve”. Đây là propose-then-commit (đề xuất rồi thực thi): hành động đề xuất được lưu bền vững cùng idempotency key (khóa chống thực thi lặp); trình cho người duyệt kèm mục đích, data lineage (nguồn gốc dữ liệu), quyền bị tác động, blast radius (phạm vi ảnh hưởng) và kế hoạch hoàn tác; chỉ thực thi sau khi được xác nhận rõ ràng; rồi kiểm chứng tác động đã xảy ra. `interrupt()` của LangGraph cùng cơ chế lưu checkpoint vào PostgreSQL, `RequestInfoEvent` của Microsoft Agent Framework và `waitForApproval()` của Cloudflare đều theo mô hình này. Lỗi điển hình là phê duyệt qua loa: bấm “Approve?” mà không xem xét. Biện pháp được ghi nhận là challenge-and-response (hỏi đáp xác nhận) với checklist rõ ràng.

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Giai đoạn 15, Bài 12 (Thực thi bền vững), Giai đoạn 15, Bài 14 (Cơ chế chặn)
**Time:** ~60 phút

## Vấn đề

Agent thực hiện một hành động. Người dùng phải quyết định: duyệt hay không. Nếu quyết định tức thì, đó có lẽ không phải xem xét. Nếu xem xét có cấu trúc, việc này chậm hơn nhưng đáng tin hơn. Câu hỏi kỹ thuật là làm sao để xem xét có cấu trúc trở thành cách dễ làm nhất.

Mô hình HITL thời 2023 là lời hỏi đồng bộ: “Agent muốn gửi email cho X với nội dung Y — duyệt chứ?” Người dùng bấm Approve. Ai cũng thấy hệ thống an toàn. Thực tế, người dùng thường duyệt qua loa: họ duyệt nhanh, quyết định duyệt ít phản ánh rủi ro, và khi agent làm sai, nhật ký chỉ ghi hàng loạt lần duyệt mà người dùng không nhớ.

Mô hình năm 2026 — propose-then-commit — đặt HITL trên nền tảng lưu trữ bền vững, gắn metadata có cấu trúc và yêu cầu xác nhận để thực thi. Mọi SDK agent được quản lý đều có phiên bản riêng: `interrupt()` của LangGraph, `RequestInfoEvent` của Microsoft Agent Framework, `waitForApproval()` của Cloudflare. Tên API khác nhau; mô hình thì không.

## Khái niệm

### Máy trạng thái propose-then-commit

1. **Đề xuất.** Agent tạo hành động đề xuất và lưu vào kho bền vững (PostgreSQL, Redis, Durable Object). Đề xuất gồm:
   - mục đích (vì sao agent làm việc này)
   - data lineage (nguồn nào dẫn tới đề xuất)
   - quyền bị tác động (scope / tệp / endpoint nào)
   - blast radius (tình huống xấu nhất là gì)
   - kế hoạch hoàn tác (sau khi thực thi, hoàn tác thế nào)
   - idempotency key (duy nhất cho mỗi đề xuất; gửi lại sẽ nhận cùng bản ghi)
2. **Trình duyệt.** Người duyệt thấy đề xuất cùng toàn bộ metadata. Người duyệt là con người (không phải agent tự duyệt).
3. **Thực thi.** Sau khi được xác nhận rõ ràng, hành động được thực hiện.
4. **Kiểm chứng.** Sau khi thực hiện, hệ thống đọc lại và xác nhận tác động. Nếu kiểm chứng thất bại, hệ thống ở trạng thái lỗi đã biết và phát cảnh báo.

### Idempotency key

Không có idempotency key, thử lại sau lỗi tạm thời có thể thực hiện hành động đã duyệt hai lần. Ví dụ: người dùng duyệt “chuyển $100 từ A sang B”. Mạng chập chờn. Workflow thử lại. Người dùng chỉ duyệt một lần nhưng tiền được chuyển hai lần. Idempotency key ràng buộc lần duyệt với một tác động duy nhất; lần thực hiện thứ hai không làm gì.

Đây là mô hình chống thực thi lặp mà API của Stripe và AWS dùng. Tài liệu Microsoft Agent Framework nêu rõ việc áp dụng nó cho phê duyệt hành động của agent.

### Tính bền vững: vì sao phê duyệt tồn tại lâu hơn tiến trình

Trạng thái chờ phê duyệt không thuộc sở hữu của agent. Workflow được tạm dừng (Bài 12). Khi có phê duyệt, workflow tiếp tục đúng tại điểm đó. Vì vậy LangGraph kết hợp `interrupt()` với lưu checkpoint vào PostgreSQL, thay vì chỉ lưu trong bộ nhớ — phê duyệt đến sau hai ngày vẫn tìm thấy workflow nguyên vẹn.

### Phê duyệt qua loa và biện pháp challenge-and-response

Giao diện HITL mặc định (nút “Approve” / “Reject”) khiến người dùng duyệt nhanh mà không thực sự xem xét. Biện pháp được ghi nhận là checklist challenge-and-response: phải trả lời xác nhận các câu hỏi cụ thể trước khi nút Approve được bật. Ví dụ:

- “Bạn hiểu hành động này tác động đến tài nguyên nào chứ? [ ]”
- “Bạn đã xác minh blast radius ở mức chấp nhận được chưa? [ ]”
- “Bạn có kế hoạch hoàn tác nếu hành động thất bại không? [ ]”

Đây không phải thủ tục vì thủ tục — mà là cách buộc người duyệt cân nhắc. Nếu không thể đánh dấu các ô, họ sẽ hỏi thêm (chuyển cấp) hoặc từ chối (mặc định an toàn). Nghiên cứu an toàn agent của Anthropic nêu rõ HITL dựa trên checklist là biện pháp chống phê duyệt qua loa.

### Hành động nào có hệ quả đáng kể

Không phải hành động nào cũng cần propose-then-commit. Hướng dẫn năm 2026:

- **Hành động có hệ quả đáng kể** (luôn cần HITL): ghi không thể đảo ngược, giao dịch tài chính, liên lạc ra ngoài, thay đổi cơ sở dữ liệu production, thao tác phá hủy trên hệ thống tệp.
- **Hành động có thể đảo ngược** (đôi khi cần HITL): sửa tệp cục bộ, thay đổi môi trường staging, ghi có cách hoàn tác rõ ràng.
- **Đọc và kiểm tra** (không cần HITL): đọc tệp, liệt kê tài nguyên, gọi API chỉ đọc.

### Kiểm chứng sau hành động

“Lệnh thực thi đã chạy” không đồng nghĩa “tác động đã xảy ra”. Mạng bị phân đoạn và race condition có thể khiến workflow tưởng đã thành công dù backend chưa lưu thay đổi. Bước kiểm chứng đọc lại tài nguyên đích sau khi thực thi để xác nhận. Đây cũng là mô hình của giao dịch cơ sở dữ liệu với mệnh đề `RETURNING`, hoặc AWS `GetObject` sau `PutObject`.

### EU AI Act Article 14

Article 14 yêu cầu giám sát hữu hiệu của con người đối với hệ thống AI rủi ro cao tại EU. “Hữu hiệu” không phải lời trang trí. Ngôn ngữ quy định loại trừ rõ các kiểu phê duyệt qua loa. Theo tài liệu tuân thủ Microsoft Agent Governance Toolkit, propose-then-commit cùng challenge-and-response là mô hình đáp ứng được việc thẩm định theo Article 14.

```figure
mx-propose-then-commit
```

## Áp dụng

`code/main.py` triển khai máy trạng thái propose-then-commit bằng Python thư viện chuẩn. Kho lưu trữ bền vững là tệp JSON. Idempotency key là giá trị hash của (thread_id, action_signature). Chương trình mô phỏng ba trường hợp: duyệt suôn sẻ, thử lại sau lỗi tạm thời (không được thực hiện hai lần), và phê duyệt qua loa mặc định so với quy trình challenge-and-response.

## Đóng gói

`outputs/skill-hitl-design.md` đánh giá một workflow HITL đề xuất theo mô hình propose-then-commit và chỉ ra phần thiếu về metadata, chống thực thi lặp, kiểm chứng hoặc challenge-and-response.

## Bài tập

1. Chạy `code/main.py`. Xác nhận rằng khi thử lại một đề xuất đã duyệt, hệ thống dùng bản ghi bền vững và không thực hiện lại. Sau đó thêm timestamp vào idempotency key và cho thấy lần thử lại thực hiện hành động hai lần.

2. Thêm trường `rollback` vào bản ghi đề xuất. Mô phỏng một lần thực hiện mà bước kiểm chứng thất bại. Cho thấy việc hoàn tác tự động được kích hoạt.

3. Đọc tài liệu `RequestInfoEvent` của Microsoft Agent Framework. Tìm một trường metadata có trong API nhưng thiếu ở bộ máy mẫu. Thêm trường đó và giải thích nó giúp phòng tránh điều gì.

4. Thiết kế checklist challenge-and-response cho một hành động cụ thể (ví dụ: “đăng lên tài khoản Twitter công khai”). Người duyệt phải trả lời ba câu hỏi nào? Vì sao chọn ba câu đó?

5. Chọn một trường hợp mà lời hỏi đồng bộ “Approve?” là đủ (không cần kho bền vững). Giải thích vì sao và nêu loại rủi ro bạn chấp nhận.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Propose-then-commit | “Phê duyệt hai giai đoạn” | Lưu đề xuất + xác nhận thực thi + kiểm chứng |
| Idempotency key | “Token an toàn khi thử lại” | Duy nhất cho mỗi đề xuất; lần thực hiện thứ hai không làm gì |
| Data lineage | “Nó đến từ đâu” | Nội dung nguồn cụ thể dẫn tới đề xuất |
| Blast radius | “Tình huống xấu nhất” | Phạm vi ảnh hưởng nếu hành động sai |
| Rubber-stamp | “Duyệt nhanh” | Bấm “Approve” mà không thực sự xem xét |
| Challenge-and-response | “Checklist buộc cân nhắc” | Người duyệt phải xác nhận rõ các câu hỏi cụ thể |
| RequestInfoEvent | “Cấu phần MS Agent Framework” | Yêu cầu HITL bền vững với metadata có cấu trúc |
| `interrupt()` / `waitForApproval()` | “Cấu phần framework” | Các cách tương đương của LangGraph / Cloudflare cho cùng mô hình |

## Đọc thêm

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, phê duyệt bền vững.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` và Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL giúp giảm rủi ro dài hạn.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — nền tảng quy định cho hệ thống rủi ro cao.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — cách nhìn theo hiến chương về giám sát.

---

[← Bài trước: Công tắc ngắt, cầu dao và token mồi](../../14-kill-switches-canaries/docs/vi.md) · [Bài sau: Checkpoint và rollback →](../../16-checkpoints-rollback/docs/vi.md)
