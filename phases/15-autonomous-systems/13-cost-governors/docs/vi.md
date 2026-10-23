# Ngân sách hành động, giới hạn vòng lặp và kiểm soát chi phí

> Chi phí LLM hằng tháng của một agent thương mại điện tử cỡ vừa tăng từ $1,200 lên $4,800 sau khi nhóm bật kỹ năng "order-tracking". Đây không phải lỗi giá. Agent đã tìm ra một vòng lặp mới và liên tục tiêu tiền trong đó. Agent Governance Toolkit của Microsoft (ngày 2/4/2026) hệ thống hóa cách phòng vệ: `max_tokens` cho mỗi yêu cầu, ngân sách token và tiền cho mỗi tác vụ, giới hạn theo ngày/tháng, giới hạn vòng lặp, định tuyến theo tầng mô hình, cache prompt, giới hạn cửa sổ ngữ cảnh, điểm kiểm tra HITL trước hành động tốn kém, và công tắc dừng khi vượt ngân sách. Claude Code Agent SDK của Anthropic cung cấp cùng cơ chế dưới tên khác. Giới hạn tốc độ chi tiêu — chẳng hạn cắt quyền truy cập khi tiêu >$50 trong 10 phút — phát hiện vòng lặp nhanh hơn giới hạn tháng.

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Giai đoạn 15, Bài 10 (Chế độ quyền), Giai đoạn 15, Bài 12 (Thực thi bền vững)
**Time:** ~60 phút

## Vấn đề

Agent tự chủ tiêu tiền thật ở mỗi lượt. Chatbot trả lời sai chỉ tạo ra một câu trả lời tệ; agent mắc vòng lặp tạo ra hóa đơn. Ngành gọi kiểu lỗi này là "Denial of Wallet" (cạn ví): agent tiếp tục suy luận, gọi công cụ và phát sinh phí vì không có cơ chế dừng được thiết kế sẵn.

Cách khắc phục không phải một con số, mà là nhiều lớp giới hạn theo thời gian và phạm vi: mỗi yêu cầu, mỗi tác vụ, mỗi giờ, mỗi ngày, mỗi tháng. Hệ thống tốt chặn vòng lặp mất kiểm soát trong vài phút, rò rỉ chi phí chậm trong vài giờ và bản phát hành lỗi trong một ngày. Nó cũng giữ agent tự chủ chạy dài hạn trong ngân sách.

Đây là bài học kỹ thuật: phép tính rất đơn giản, nhưng các nhóm thường thiếu kỷ luật thực hiện. Mọi giới hạn dưới đây đều được nêu trong Microsoft Agent Governance Toolkit hoặc tài liệu Anthropic Claude Code Agent SDK.

## Khái niệm

### Các lớp kiểm soát chi phí

1. **`max_tokens` cho mỗi yêu cầu.** Đơn giản: ngăn một lần gọi tạo phần hoàn thành dài không giới hạn.
2. **Ngân sách token cho mỗi tác vụ.** Không vượt N token trong toàn bộ lần chạy. Dừng hẳn khi chạm giới hạn.
3. **Ngân sách tiền cho mỗi tác vụ.** Tương tự ngân sách token nhưng tính bằng tiền. Claude Code dùng `max_budget_usd`.
4. **Giới hạn lượt gọi từng công cụ.** Không quá N lượt gọi `WebFetch`, N lượt gọi `shell_exec`, v.v.
5. **Giới hạn vòng lặp (`max_turns`).** Giới hạn tổng lượt trong vòng lặp agent; ngăn vòng suy luận vô hạn.
6. **Giới hạn mỗi phút / giờ / ngày / tháng.** Áp dụng trên cửa sổ trượt. Phát hiện rò rỉ ở nhiều thang thời gian.
7. **Giới hạn tốc độ chi tiêu.** Chẳng hạn, "nếu chi tiêu vượt $50 trong 10 phút, cắt quyền truy cập." Chặn vòng lặp đốt tiền trước khi giới hạn tháng có hiệu lực.
8. **Định tuyến theo tầng mô hình.** Mặc định dùng mô hình nhỏ; chỉ chuyển sang mô hình lớn khi bộ phân loại xác định tác vụ cần thiết.
9. **Cache prompt.** Provider lưu system prompt và ngữ cảnh ổn định; chi phí token khi gửi lại gần bằng không.
10. **Giới hạn cửa sổ ngữ cảnh.** Nén / tóm tắt để giữ ngữ cảnh đang dùng dưới ngưỡng; trực tiếp giảm chi phí token.
11. **Điểm kiểm tra HITL trước hành động tốn kém.** Yêu cầu người phê duyệt trước hành động vốn đắt đỏ (lượt gọi công cụ dài, tải dữ liệu lớn, nâng cấp lên mô hình đắt tiền).
12. **Công tắc dừng khi vượt ngân sách.** Phiên chấm dứt khi bất kỳ giới hạn nào kích hoạt. Hệ thống ghi lại giới hạn đó; muốn chạy tiếp phải qua quy trình bật lại riêng.

### Vì sao cần nhiều lớp, không chỉ một giới hạn

Một giới hạn tháng chỉ chặn agent mất kiểm soát sau khi tiền đã cạn. Giới hạn cho mỗi yêu cầu không chặn được chi phí ở cấp phiên. Mỗi kiểu lỗi cần một thang thời gian khác nhau:

- **Vòng lặp mất kiểm soát** (agent kẹt trong chu kỳ thử lại 5 giây): giới hạn tốc độ chi tiêu sẽ chặn.
- **Rò rỉ chậm** (agent làm ~2x lượng việc dự kiến mỗi tác vụ): giới hạn ngày sẽ chặn.
- **Bản phát hành lỗi** (phiên bản mới dùng 5x token): giới hạn tuần / tháng sẽ chặn.
- **Nhu cầu tăng thật** (không phải lỗi): giới hạn giờ / ngày sẽ ghi nhận rõ trong log.

### Các giới hạn ngân sách của hệ điều phối

Claude Code Agent SDK cung cấp các mục sau (theo tài liệu công khai):

- `max_turns` — giới hạn vòng lặp.
- `max_budget_usd` — giới hạn tiền; phiên chấm dứt khi vượt mức.
- `allowed_tools` / `disallowed_tools` — danh sách công cụ được phép và bị cấm.
- Các điểm hook trước khi dùng công cụ để tự theo dõi chi phí.

Kết hợp với các cấp chế độ quyền (Bài 10). Phiên `autoMode` không có `max_budget_usd` là quyền tự chủ thiếu kiểm soát. Anthropic nêu rõ Auto Mode cần kiểm soát ngân sách; bộ phân loại không thay thế việc kiểm soát chi phí.

### EU AI Act, OWASP Agentic Top 10

Microsoft's Agent Governance Toolkit bao quát OWASP Agentic Top 10 và yêu cầu tại EU AI Act Article 14 (giám sát của con người). Khi triển khai production ở EU, ghi log và thực thi giới hạn là bắt buộc.

### Trường hợp $1,200 → $4,800 được ghi nhận

Trường hợp thực trong tài liệu Microsoft: chi phí hằng tháng của một agent thương mại điện tử tăng gấp ba sau khi thêm công cụ mới. Công cụ cho phép agent liên tục kiểm tra trạng thái đơn hàng trong mọi phiên. Không có phát hiện vòng lặp, giới hạn riêng cho công cụ hay cảnh báo khi chi phí tăng qua từng tuần. Cách khắc phục là thêm giới hạn cho công cụ và cảnh báo tăng trưởng hằng ngày. Đây là mẫu chung: mỗi công cụ mới mở ra một vòng lặp tiềm ẩn; mỗi công cụ cần giới hạn và cảnh báo riêng.

```figure
cost-governor-stack
```

## Áp dụng

`code/main.py` mô phỏng lần chạy agent có và không có nhiều lớp kiểm soát chi phí. Sau một số lượt, agent mô phỏng chuyển dần sang vòng lặp kiểm tra trạng thái; các lớp giới hạn chặn nó trong cửa sổ tốc độ chi tiêu, còn một giới hạn tháng đơn lẻ phải vài ngày sau mới kích hoạt.

## Đóng gói

`outputs/skill-agent-budget-audit.md` kiểm tra các lớp kiểm soát chi phí của một bản triển khai agent đề xuất và đánh dấu lớp còn thiếu.

## Bài tập

1. Chạy `code/main.py`. Xác nhận giới hạn tốc độ chi tiêu kích hoạt trước giới hạn vòng lặp khi agent liên tục kiểm tra trạng thái. Sau đó tắt giới hạn tốc độ và đo agent "tiêu" bao nhiêu trước khi giới hạn vòng lặp chặn lại.

2. Thiết kế các giới hạn theo công cụ cho một browser agent (Bài 11). Công cụ nào cần giới hạn chặt nhất? Công cụ nào có thể chạy không giới hạn mà không rủi ro?

3. Đọc tài liệu Microsoft Agent Governance Toolkit. Liệt kê mọi loại giới hạn được nêu. Ghép từng loại với một kiểu lỗi (vòng lặp mất kiểm soát, rò rỉ chậm, bản phát hành lỗi, nhu cầu tăng).

4. Ước tính chi phí chạy không giám sát qua đêm cho một tác vụ thực tế (chẳng hạn "phân loại 50 issue trong repo"). Đặt `max_budget_usd` bằng 2x mức ước tính. Giải thích vì sao chọn 2x.

5. `max_budget_usd` của Claude Code kích hoạt theo tổng chi phí phiên. Thiết kế thêm giới hạn tốc độ chi tiêu để thực thi bên ngoài. Điều gì kích hoạt việc cắt quyền truy cập, và quy trình bật lại ra sao?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Denial of Wallet | "Hóa đơn mất kiểm soát" | Vòng lặp agent phát sinh chi phí mà không có giới hạn chặn lại |
| max_tokens | "Giới hạn mỗi yêu cầu" | Mức trần độ dài phần hoàn thành của một lần gọi |
| max_turns | "Giới hạn vòng lặp" | Mức trần số lượt lặp của agent trong một phiên |
| max_budget_usd | "Công tắc dừng theo tiền" | Giới hạn chi phí phiên; vượt mức sẽ chấm dứt phiên |
| Velocity limit | "Giới hạn tốc độ" | Giới hạn chi tiêu trong cửa sổ thời gian ngắn (chẳng hạn $50 / 10 min) |
| Tiered routing | "Dùng mô hình nhỏ trước" | Mặc định dùng mô hình rẻ; chỉ nâng cấp khi bộ phân loại xác định cần thiết |
| Prompt caching | "Cache system prompt" | Cache phía provider giảm chi phí token khi gửi lại xuống gần bằng không |
| HITL checkpoint | "Cổng phê duyệt của con người" | Cần người phê duyệt trước hành động tốn kém |

## Đọc thêm

- [Anthropic Claude Code Agent SDK — vòng lặp agent và ngân sách](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, danh sách công cụ được phép.
- [Microsoft Agent Framework — con người trong vòng kiểm soát và quản trị](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các điểm kiểm tra chi phí.
- [Anthropic — tổng quan Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — kiểm soát chi phí phía provider.
- [Anthropic — Prompt caching (tài liệu Claude API)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — cơ chế cache.
- [Anthropic — đo lường quyền tự chủ của agent trong thực tế](https://www.anthropic.com/research/measuring-agent-autonomy) — đặc điểm chi phí của agent chạy dài hạn.

---

[← Bài trước: Agent chạy nền dài hạn: Thực thi bền vững](../../12-durable-execution/docs/vi.md) · [Bài sau: Công tắc ngắt, cầu dao và token mồi →](../../14-kill-switches-canaries/docs/vi.md)
