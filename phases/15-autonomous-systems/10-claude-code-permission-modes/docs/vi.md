# Chế độ cấp quyền cho agent tự chủ

> Thang cấp quyền (permission ladder) — các mức tự chủ tăng dần, từ xét duyệt từng hành động đến duyệt tất cả — là cách harness kiểm soát việc agent tự chủ được làm mà không cần hỏi. Claude Code, ví dụ xuyên suốt bài này, có sáu chế độ: "plan" yêu cầu duyệt trước mọi hành động, "default" (ghi là "Manual" trong UI) chỉ hỏi với hành động rủi ro, "acceptEdits" tự duyệt việc ghi file nhưng vẫn hỏi trước khi chạy lệnh shell, còn "bypassPermissions" duyệt tất cả. Auto Mode — chế độ cấp quyền `auto` — thay việc duyệt từng hành động bằng một mô hình phân loại riêng: mô hình xét từng hành động trước khi chạy và chặn hành động vượt quá yêu cầu. Giới hạn hành động được áp qua `max_turns` và `max_budget_usd`. Việc có dùng được `auto` hay không phụ thuộc vào gói dịch vụ, việc tổ chức bật chế độ này, mô hình và nhà cung cấp — và Anthropic nói rõ rằng chỉ riêng mô hình phân loại là chưa đủ.

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Giai đoạn 15, Bài 01 (Agent chạy dài hạn), Giai đoạn 15, Bài 09 (Bức tranh agent viết code)
**Time:** ~45 phút

## Vấn đề

Agent viết code tự chủ trên máy bạn là một loại rủi ro bảo mật riêng. Bề mặt tấn công (attack surface) gồm mọi thứ agent có thể truy cập — hệ thống file, mạng, thông tin xác thực, clipboard, mọi tab trình duyệt và terminal đang mở. Bruce Schneier cùng những người khác đã công khai cảnh báo: agent dùng máy tính không phải bản "cập nhật tính năng" của chatbot, mà là công cụ mới với rủi ro mới.

Hệ thống cấp quyền của Claude Code là câu trả lời của Anthropic. Thay vì một công tắc "tự chủ / không tự chủ", hệ thống có sáu chế độ trên thang năng lực: plan → default → acceptEdits → … → bypassPermissions. Mỗi chế độ đánh đổi khác nhau giữa tốc độ và việc xét duyệt từng hành động. Auto Mode (tháng 3/2026) thêm một mô hình phân loại riêng, đưa việc phê duyệt ra khỏi luồng chờ của người dùng: mô hình xét từng hành động trước khi chạy và chặn hành động vượt quá yêu cầu.

Câu hỏi kỹ thuật là: hệ thống này phát hiện gì, bỏ sót gì, và công việc cụ thể thực sự cần chế độ nào?

## Khái niệm

### Sáu chế độ cấp quyền

| Chế độ | Cách hoạt động | Khi nào dùng |
|---|---|---|
| `plan` | Agent đề xuất kế hoạch; người dùng duyệt toàn bộ kế hoạch; từng hành động được xét trước khi chạy | Công việc chưa quen; code gần môi trường production; lần đầu dùng agent trên một repo |
| `default` | Ghi là "Manual" trong UI. Agent thực hiện hành động; hỏi người dùng với mọi hành động "rủi ro" (chạy lệnh shell, thao tác phá huỷ, gọi mạng) | Hầu hết phiên viết code tương tác |
| `acceptEdits` | Tự duyệt việc ghi file; vẫn hỏi trước khi chạy lệnh shell hoặc gọi mạng | Đợt tái cấu trúc trên nhiều file |
| `auto` | Mô hình phân loại riêng xét từng hành động trước khi chạy; chặn hành động vượt quá yêu cầu | Lượt chạy dài hạn không có người giám sát trong workspace có giới hạn |
| `dontAsk` | Không bao giờ hỏi; từ chối hành động chưa được quy tắc cấp quyền duyệt trước | Sandbox dùng một lần, tác vụ CI, script nghiên cứu |
| `bypassPermissions` | Duyệt tất cả | Tài liệu ghi "chỉ dùng trong container tạm mà bạn sẵn sàng bỏ đi" |

(Tên trên khớp tài liệu Claude Code công khai; UI ghi `default` là "Manual".)

### Auto Mode trong một trang

Auto Mode (ra mắt ngày 24/3/2026) là chế độ cấp quyền đầu tiên giao việc duyệt từng hành động cho mô hình. Cấu trúc:

1. **Một mô hình phân loại riêng.** Xét từng hành động đề xuất trước khi chạy, dựa trên nhiệm vụ đã nêu và trạng thái hiện tại của phiên; chặn hành động vượt quá yêu cầu. Hành động bị chặn sẽ chuyển cho người dùng.
2. **Điều kiện sử dụng.** Việc `auto` có được cung cấp hay không phụ thuộc vào gói dịch vụ, việc tổ chức bật chế độ này, mô hình và nhà cung cấp.

Kiểm soát ngân sách hoạt động cùng mô hình phân loại:

- `max_turns` — tổng số vòng lặp trong một phiên.
- `max_budget_usd` — mức trần chi phí khiến phiên dừng khi chạm tới.
- giới hạn số hành động theo công cụ (không quá N lần gọi `WebFetch`, v.v.).

### Hệ thống phát hiện được gì

- Prompt injection (chèn chỉ dẫn độc hại) đơn giản trong đầu vào công cụ, khi chỉ dẫn được chèn dẫn đến kiểu hành động đã biết là rủi ro.
- Vòng lặp công cụ lặp lại — mô hình phân loại thấy hành động N+1 gần như giống hành động N, năm lần liên tiếp.
- Lệnh shell rõ ràng ngoài phạm vi trong phiên vốn chỉ được sửa file.

### Hệ thống có thể bỏ sót gì

- **Prompt injection tinh vi** làm đổi hành vi mà không tạo ra hành động riêng lẻ nào bị gắn cờ. Prompt injection gián tiếp không phải lỗ hổng có thể vá hoàn toàn (người đứng đầu nhóm preparedness của OpenAI, 2025, nói về agent trình duyệt — xem Bài 11).
- **Hành vi sai ở cấp độ ngữ nghĩa.** Từng hành động có thể trông an toàn trong khi cả chuỗi gây hại. Mô hình phân loại xét hành động; nó không tự suy luận lại ý định người dùng.
- **Rò rỉ dữ liệu qua kênh hợp lệ.** Ghi dữ liệu vào file bạn sở hữu rồi `git push` lên repo công khai là chuỗi hành động được phép, nhưng vấn đề nằm ở cách chúng kết hợp.

### Góc nhìn về bản thử nghiệm nghiên cứu

Anthropic phát hành Auto Mode dưới dạng bản thử nghiệm nghiên cứu. Tài liệu nói rõ mô hình phân loại là một lớp bảo vệ, không phải giải pháp hoàn chỉnh: người dùng cần kết hợp Auto Mode với giới hạn ngân sách, danh sách cho phép (allowlist), workspace cách ly và việc kiểm tra chuỗi hành động (Bài 12–16). Cách gọi bản thử nghiệm cũng phản ánh khoảng cách đã được ghi nhận giữa đánh giá và triển khai (Bài 1) — mô hình phân loại vượt qua bài đánh giá offline vẫn có thể hành xử khác trong phiên thực, khi ngữ cảnh người dùng không rõ ràng.

### Chọn bậc nào trong quy trình làm việc

- Công việc chưa quen: bắt đầu với `plan`. Đọc kế hoạch ít tốn công hơn hoàn tác một lượt chạy sai.
- Tái cấu trúc đã rõ: `acceptEdits` giúp bớt nhiều lần bấm xác nhận.
- Chạy nền không giám sát: chỉ dùng `auto` trong workspace có phạm vi ảnh hưởng bạn đã đo được (không có thông tin xác thực, không gắn môi trường production, không có kết nối ra ngoài mà bạn chưa cho phép).
- Container tạm: `dontAsk` / `bypassPermissions` chỉ chấp nhận được khi cả container lẫn thông tin xác thực trong đó đều có thể bỏ đi.

```figure
autonomy-oversight
```

## Áp dụng

`code/main.py` mô phỏng bộ phân loại xét duyệt hành động bằng pipeline hai giai đoạn — đây là cách đơn giản hoá để giảng dạy; chế độ `auto` thực dùng mô hình phân loại riêng, không có cam kết hai giai đoạn được ghi trong tài liệu. Giai đoạn 1 dùng quy tắc từ khoá nhanh trên hành động đề xuất; Giai đoạn 2 xét chậm hơn bằng nhiều quy tắc. Chương trình đưa vào một chuỗi hành động giả lập ngắn (hành động an toàn, một lần thử prompt injection, một vòng lặp lặp lại) và cho thấy bộ phân loại phát hiện hay bỏ sót ở đâu.

## Đóng gói

`outputs/skill-permission-mode-picker.md` ghép mô tả công việc với chế độ cấp quyền, mức trần ngân sách và cách ly cần thiết.

## Bài tập

1. Chạy `code/main.py`. Loại hành động giả lập nào không bao giờ bị Giai đoạn 1 gắn cờ nhưng luôn bị Giai đoạn 2 phát hiện? Loại nào cả hai đều bỏ sót?

2. Mở rộng bộ quy tắc Giai đoạn 1 để phát hiện một mẫu xấu cụ thể đã biết (ví dụ `curl $ATTACKER/exfil`). Đo tỷ lệ báo động nhầm trên mẫu hành động lành tính.

3. Đọc tài liệu "How the agent loop works" của Anthropic. Liệt kê mọi trạng thái bên ngoài mà agent mặc định tác động tới trong chế độ `default`. Trước khi chạy `auto` không giám sát, bạn cần kiểm soát riêng những trạng thái nào?

4. Thiết kế ngân sách chạy không giám sát trong 24 giờ: `max_turns`, `max_budget_usd`, giới hạn theo công cụ, danh sách cho phép. Giải thích từng con số.

5. Mô tả một chuỗi mà mô hình phân loại duyệt mọi hành động riêng lẻ, nhưng hành vi tổng thể vẫn lệch mục tiêu. (Bài 14 trình bày cách công tắc dừng khẩn cấp và canary token xử lý vấn đề này.)

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Permission mode | "Agent làm được bao nhiêu" | Một trong sáu chính sách có tên, kiểm soát việc duyệt từng hành động |
| plan mode | "Hỏi trước mọi việc" | Agent viết kế hoạch; người dùng duyệt trước khi thực hiện |
| acceptEdits | "Cho phép ghi file" | Tự duyệt việc ghi file; vẫn hỏi trước khi chạy lệnh shell |
| auto | "Tự động phê duyệt" | Mô hình phân loại riêng xét từng hành động; chặn hành động vượt quá yêu cầu |
| bypassPermissions | "Làm mọi thứ, bất chấp rủi ro" | Duyệt tất cả; dành cho container tạm |
| Stage 1 (simulator) | "Kiểm tra nhanh bằng từ khoá" | Quy tắc đơn giản xét hành động đề xuất trong `code/main.py` |
| Stage 2 (simulator) | "Xét duyệt sâu" | Bộ xét duyệt nhiều quy tắc, chậm hơn, cho hành động bị gắn cờ trong `code/main.py` |
| Research preview | "Chưa phát hành rộng rãi" | Cách Anthropic gọi tính năng có các kiểu lỗi vẫn đang được tìm hiểu |

## Đọc thêm

- [Anthropic — Cách vòng lặp agent hoạt động](https://code.claude.com/docs/en/agent-sdk/agent-loop) — chế độ cấp quyền, ngân sách, định dạng hành động.
- [Anthropic — Tổng quan Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — mô hình thực thi của dịch vụ được quản lý.
- [Anthropic — Trang sản phẩm Claude Code](https://www.anthropic.com/product/claude-code) — các tính năng và thông báo về Auto Mode.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — lớp lập luận định hình đánh giá của mô hình phân loại.
- [Anthropic — Đo mức tự chủ của agent trong thực tế](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về thiết kế cấp quyền cho lượt chạy dài hạn.

---

[← Bài trước: Toàn cảnh agent lập trình tự chủ (2026)](../../09-coding-agent-landscape/docs/vi.md) · [Bài sau: Tác tử trình duyệt và tác vụ web dài hạn →](../../11-browser-agents/docs/vi.md)
