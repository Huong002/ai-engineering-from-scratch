# Checkpoint và rollback

> Mọi lần chuyển trạng thái đồ thị đều được lưu. Khi worker gặp sự cố, lease hết hạn và worker khác tiếp tục từ checkpoint mới nhất. Cloudflare Durable Objects giữ trạng thái qua nhiều giờ hoặc nhiều tuần. Propose-then-commit (Bài 15) xác định kế hoạch rollback cho từng hành động. Kiểm chứng sau hành động khép kín quy trình. EU AI Act Article 14 yêu cầu giám sát hiệu quả của con người với hệ thống rủi ro cao — trên thực tế, checkpoint phải truy vấn được, rollback phải được diễn tập và dấu vết kiểm toán phải tồn tại sau khi deploy. Lỗi nguy hiểm nhất: thiếu idempotency key và kiểm tra tiền điều kiện, một lần thử lại sau lỗi tạm thời có thể thực thi hai lần hành động đã duyệt. Kiểm chứng sau hành động giúp phát hiện lỗi này.

**Type:** Learn
**Languages:** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites:** Giai đoạn 15, Bài 12 (Thực thi bền vững), Giai đoạn 15, Bài 15 (Propose-then-commit)
**Time:** ~60 phút

## Vấn đề

Thực thi bền vững (Bài 12) giúp agent tiếp tục sau sự cố. Propose-then-commit (Bài 15) giúp kiểm toán hành động đã duyệt. Bài này kết hợp cả hai: điều gì xảy ra khi hành động đã duyệt thực thi một phần, gặp sự cố rồi tiếp tục? Khi nào rollback chạy, và trên trạng thái nào?

Các hệ thống thực tế triển khai khác nhau:

- **LangGraph** lưu checkpoint sau mỗi lần chuyển trạng thái đồ thị vào PostgreSQL. Khi worker gặp sự cố, lease được giải phóng và worker khác tiếp tục từ checkpoint mới nhất. Workflow tạm dừng tại `interrupt()`, và lần tạm dừng đó cũng được lưu.
- **Cloudflare Durable Objects** giữ trạng thái theo từng khóa qua nhiều giờ hoặc nhiều tuần. Đặt xử lý cùng nơi lưu trữ hành động đã duyệt.
- **Microsoft Agent Framework** cung cấp các thành phần `Checkpoint` trong workflow API; phát lại kết hợp tính idempotency xử lý các lần thử lại.

Trong mọi trường hợp, tổ hợp thực sự hiệu quả là: idempotency key (khóa chống thực thi trùng) + kiểm tra tiền điều kiện (trạng thái vẫn khớp lúc duyệt) + kiểm chứng sau hành động (tác động thực sự xảy ra) + rollback khi kiểm chứng thất bại.

## Khái niệm

### Lưu mọi lần chuyển trạng thái

Một lần chuyển trạng thái đồ thị là bất kỳ bước nào đưa workflow từ trạng thái có tên này sang trạng thái có tên khác. Cách làm đơn giản chỉ lưu tại vài điểm commit; hệ thống production lưu mọi lần chuyển. Chi phí (thêm vài lần ghi) nhỏ so với lợi ích về độ tin cậy (có thể phát lại từ mọi điểm, phục hồi lease chính xác).

### Phục hồi lease

Khi worker gặp sự cố, workflow không mất; lease (quyền tạm thời xác nhận worker này đang chạy tác vụ) chỉ hết hạn. Worker khác lấy checkpoint mới nhất và tiếp tục. Cơ chế lease giúp hệ thống production chịu được các đợt deploy cuốn chiếu mà không mất việc đang chạy.

### Idempotency và tiền điều kiện

Chỉ idempotency là chưa đủ. Xét workflow được duyệt để “chuyển $100 từ A sang B khi số dư > $1000”. Workflow bắt đầu commit, gặp sự cố giữa chừng rồi tiếp tục. Nếu chỉ kiểm tra idempotency key, giao dịch chạy một lần (đúng). Nhưng nếu giữa lúc gặp sự cố và tiếp tục, số dư A giảm còn $500 do workflow khác thì sao? Kiểm tra idempotency vẫn qua; tiền điều kiện thì không. Thiếu kiểm tra tiền điều kiện, ta tạo ra khoản thấu chi.

Mọi hành động có hệ quả đều cần cả hai:

- **Idempotency key**: ngăn thực thi trùng.
- **Kiểm tra tiền điều kiện**: xác nhận trạng thái vẫn phù hợp với điều đã duyệt.

### Kiểm chứng sau hành động

“Công cụ trả về 200” không phải kiểm chứng. Kiểm chứng thực sự là đọc lại trạng thái đích và xác nhận tác động đã xảy ra. Các cách thường dùng:

- Cập nhật cơ sở dữ liệu: `UPDATE ... RETURNING *`, rồi xác nhận hàng trả về khớp trạng thái dự định.
- Gửi email: kiểm tra thư mục đã gửi để tìm ID thư sau khi gửi.
- Ghi tệp: đọc lại tệp và tính hash.
- Gọi API: gọi tiếp `GET` trên tài nguyên đích.

Nếu kiểm chứng thất bại, workflow ở trạng thái được xác định là lỗi. Rollback bắt đầu.

### Kế hoạch rollback

Mọi hành động có hệ quả trong propose-then-commit (Bài 15) đều kèm kế hoạch rollback. Các loại:

- **In-band rollback (hoàn tác trực tiếp)**: đảo ngược tác động ngay (`DELETE` sau `INSERT`, `Send-correction-email` sau khi gửi).
- **Compensating transaction (giao dịch bù)**: hành động mới vô hiệu hóa hành động ban đầu (mẫu SAGA chuẩn).
- **Out-of-band rollback (hoàn tác ngoài quy trình)**: báo cho người phụ trách, tạm dừng workflow, giữ nguyên trạng thái lỗi để điều tra.

Rollback không làm gì (“không thể hoàn tác”) phải được nêu trong đề xuất. Hành động không thể rollback cần HITL chặt chẽ hơn khi commit (cơ chế hỏi-đáp ở Bài 15).

### Cách hiểu EU AI Act Article 14 khi vận hành

Article 14 yêu cầu “giám sát hiệu quả của con người” với hệ thống rủi ro cao. Khi triển khai, điều này được hiểu là:

- Kiểm toán viên truy vấn được checkpoint.
- Rollback được diễn tập (kiểm thử đầu cuối ít nhất một lần).
- Dấu vết kiểm toán tồn tại sau deploy (nơi lưu checkpoint không phải tạm thời).
- Lỗi kiểm chứng phải được cảnh báo, không chỉ âm thầm ghi log.

Workflow gặp sự cố giữa commit, tiếp tục rồi hoàn tất tác động mà không có đường kiểm chứng và rollback sẽ không đáp ứng được Article 14.

### Lỗi nguy hiểm nhất: thực thi hai lần

Sự cố production phổ biến nhất trong lĩnh vực này:

1. Hành động được duyệt, idempotency key k.
2. Commit bắt đầu, thực thi, trả về 200.
3. Workflow gặp sự cố trước khi lưu trạng thái “đã commit”.
4. Workflow tiếp tục; thấy “đã duyệt nhưng chưa commit”; thực thi lại.
5. Tác động xảy ra hai lần.

Cách phòng tránh: lưu ý định “đang thực thi” trước khi chạy, thực thi với idempotency key, rồi chỉ đánh dấu “đã commit” sau khi kiểm chứng thành công. Nếu hành động đã chạy nhưng ghi trạng thái thất bại, bạn biết cần kiểm chứng và (nếu cần) chạy lại. Nếu ghi trạng thái thành công nhưng hành động thất bại, bạn kiểm chứng rồi chạy đúng một lần qua nhánh phục hồi.

```figure
checkpoint-replay
```

## Áp dụng

`code/main.py` triển khai workflow có checkpoint, idempotency, tiền điều kiện, kiểm chứng và rollback. Chương trình mô phỏng bốn tình huống: chạy bình thường, thử lại sau sự cố (idempotency chặn trùng), tiền điều kiện không đạt (workflow dừng trước khi thực thi), kiểm chứng thất bại (rollback chạy).

## Đóng gói

`outputs/skill-rollback-rehearsal.md` thiết kế bài kiểm thử diễn tập rollback cho một workflow đề xuất và kiểm toán nơi lưu checkpoint để bảo đảm dấu vết kiểm toán được giữ lại.

## Bài tập

1. Chạy `code/main.py`. Kiểm tra bốn tình huống. Với sự cố trong lúc commit, xác nhận hành động chỉ chạy đúng một lần qua các lần thử lại.

2. Sửa cách làm “đánh dấu xong trước, rồi mới thực hiện” để việc ghi trạng thái diễn ra sau hành động. Chạy lại tình huống sự cố. Đo số hành động trùng lặp.

3. Thiết kế kế hoạch rollback cho một hành động production cụ thể (ví dụ: “đăng lên kênh Slack”). Phân loại là in-band, compensating hay out-of-band. Giải thích lựa chọn.

4. Chọn một workflow bạn biết. Xác định mọi lần chuyển trạng thái. Đánh dấu yêu cầu lưu bền vững cho từng lần (lưu / không lưu). Đếm số lần hiện bạn chưa lưu.

5. Kiểm thử diễn tập rollback: thiết kế bài kiểm thử đầu cuối chạy workflow thật, làm nó gặp sự cố và xác nhận nhánh rollback chạy. Bài kiểm thử phải xác nhận điều gì?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Checkpoint | “Điểm lưu” | Mọi lần chuyển trạng thái đồ thị được lưu vào nơi lưu trữ bền vững |
| Lease | “Quyền của worker” | Quyền tạm thời xác nhận worker đang chạy tác vụ; hết hạn khi gặp sự cố |
| Precondition | “Cổng trạng thái” | Điều kiện xác nhận trạng thái vẫn phù hợp với hành động đã duyệt |
| Post-action verify | “Kiểm tra bằng cách đọc lại” | Xác nhận tác động thực sự xảy ra trong hệ thống đích |
| In-band rollback | “Hoàn tác trực tiếp” | Đảo ngược tác động bằng thao tác ngược lại |
| Compensating transaction | “Hoàn tác SAGA” | Hành động mới vô hiệu hóa hành động ban đầu |
| Mark-as-done-first | “Thứ tự ghi trạng thái” | Lưu trạng thái đã commit trước khi commit trả về |
| Article 14 | “Giám sát của con người theo EU AI Act” | Khi vận hành: checkpoint truy vấn được, rollback đã diễn tập, dấu vết kiểm toán được |

## Đọc thêm

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các thành phần checkpoint và phục hồi lease.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects làm nền lưu trạng thái.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — cơ sở pháp lý.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — cách nhìn về độ tin cậy của workflow dài hạn.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — cấu trúc workflow cho Claude Code Routines.

---

[← Bài trước: Human-in-the-Loop: Đề xuất rồi thực thi](../../15-propose-then-commit/docs/vi.md) · [Bài sau: Constitutional AI và quyền ghi đè quy tắc →](../../17-constitutional-ai/docs/vi.md)
