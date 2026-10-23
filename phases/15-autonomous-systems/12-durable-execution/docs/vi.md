# Agent chạy nền dài hạn: Thực thi bền vững

> Agent chạy dài hạn trong môi trường production không chạy bằng `while True`. Mỗi lệnh gọi LLM trở thành một activity (tác vụ) có checkpoint (điểm lưu), cơ chế thử lại và phát lại. Tích hợp OpenAI Agents SDK của Temporal được phát hành chính thức vào tháng 3 năm 2026. Claude Code Routines (Anthropic) chạy Claude Code theo lịch mà không cần tiến trình cục bộ thường trực. Phiên có thể tạm dừng để chờ người dùng, tiếp tục sau khi triển khai và khôi phục từ checkpoint mới nhất theo `thread_id`. Đằng sau cách dùng mới là một mô hình cũ — điều phối workflow (luồng công việc) — với đầu vào mới: lệnh gọi LLM là activity không tất định, phải được phát lại một cách tất định khi khôi phục.

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Giai đoạn 15, Bài 10 (Chế độ cấp quyền), Giai đoạn 15, Bài 01 (Agent chạy dài hạn)
**Time:** ~60 phút

## Vấn đề

Xét một agent chạy bốn giờ. Nó gọi ba công cụ, hỏi người dùng hai lần và gọi LLM bốn mươi lần. Giữa chừng, máy chủ chạy nó khởi động lại. Điều gì xảy ra?

- Với vòng lặp `while True` đơn giản: mọi thứ mất hết. Lượt chạy bắt đầu lại từ đầu. Ba lệnh gọi công cụ (có tác động thực) lại được thực hiện. Người dùng lại được hỏi về những việc họ đã phê duyệt. Bốn mươi lệnh gọi LLM bị tính phí lần nữa.
- Với thực thi bền vững: lượt chạy tiếp tục từ checkpoint gần nhất. Activity đã hoàn tất không chạy lại; kết quả được phát lại từ nhật ký bền vững. Người dùng không phải phê duyệt lại. Những lệnh gọi LLM đã thực hiện không bị tính phí lần nữa.

Đây là mô hình các workflow engine đã cung cấp suốt một thập kỷ (Temporal, Cadence, Uber's Cherami). Điểm mới là lệnh gọi LLM nay cũng là một loại activity — không tất định, tốn kém, có tác động phụ — và phù hợp với mô hình này.

Chủ đề xuyên suốt bài: độ tin cậy của agent chạy dài hạn giảm theo thời gian (METR ghi nhận hiện tượng “suy giảm sau 35 phút” — tỷ lệ thành công giảm gần theo bình phương độ dài tác vụ). Thực thi bền vững cho phép chạy lâu hơn mức độ tin cậy có thể đáp ứng: thiết kế đúng thì thất bại an toàn, thiết kế sai thì không.

## Khái niệm

### Activity, workflow và phát lại

- **Workflow**: mã điều phối tất định. Quy định thứ tự activity, các nhánh và điểm chờ. Nó phải tất định để có thể phát lại từ nhật ký sự kiện mà không rẽ sang hướng bất ngờ.
- **Activity**: đơn vị công việc không tất định, có thể thất bại. Ví dụ: gọi LLM, gọi công cụ, ghi tệp, gửi yêu cầu HTTP. Đầu vào và đầu ra (khi hoàn tất) của mỗi activity đều được ghi lại.
- **Event log (nhật ký sự kiện)**: nơi lưu trữ bền vững. Mọi lần bắt đầu, hoàn tất, thất bại, thử lại activity và mọi quyết định của workflow đều được ghi lại.
- **Replay (phát lại)**: khi khôi phục, mã workflow chạy lại từ đầu; activity đã hoàn tất trả về kết quả đã ghi mà không thực thi lại. Chỉ activity chưa hoàn tất mới thực sự chạy.

Cấu trúc này giống React render lại dựa trên DOM ảo, hoặc Git dựng lại thư mục làm việc từ các commit. Tính tất định của bộ điều phối giúp việc duy trì trạng thái bền vững ít tốn kém.

### Vì sao lệnh gọi LLM phù hợp

Lệnh gọi LLM:
- Không tất định (temperature > 0; ngay cả temperature 0 cũng thay đổi giữa các phiên bản mô hình).
- Tốn kém (tiền và thời gian chờ).
- Có thể thất bại (giới hạn tốc độ, hết thời gian chờ).
- Có thể gây tác động phụ (nếu gọi công cụ).

Đây chính là đặc điểm của activity. Bọc mỗi lệnh gọi LLM thành activity giúp thử lại với thời gian chờ tăng theo cấp số nhân, lưu checkpoint qua các lần khởi động lại và tạo dấu vết có thể phát lại để gỡ lỗi.

### Checkpoint theo `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects và Claude Code Routines đều đi đến cùng một dạng API: `thread_id` (hoặc giá trị tương đương) định danh phiên; mỗi lần chuyển trạng thái được lưu vào backend (PostgreSQL mặc định, SQLite khi phát triển, Redis làm cache); khi tiếp tục, hệ thống đọc checkpoint mới nhất.

Việc chọn backend rất quan trọng:

- **PostgreSQL**: lưu bền vững, truy vấn được, không mất khi triển khai. Mặc định của LangGraph.
- **SQLite**: chỉ dành cho phát triển cục bộ; mất dữ liệu khi đổi máy chủ.
- **Redis**: nhanh nhưng dữ liệu có thể mất nếu không cấu hình AOF/snapshot.
- **Cloudflare Durable Objects**: phân tán một cách trong suốt; mỗi đối tượng gắn với một khóa duy nhất; tồn tại từ vài giờ đến vài tuần.

### Chờ người dùng như một trạng thái riêng

Quy trình đề xuất rồi xác nhận (Bài 15) cần trạng thái “chờ người dùng” được lưu bền vững. Workflow tạm dừng, hàng đợi bên ngoài giữ yêu cầu đang chờ, và việc phê duyệt cho phép tiếp tục đúng từ điểm đó. Không có cơ chế này, hệ thống chỉ có thể cố gắng duy trì trạng thái; có nó, phê duyệt gửi qua đêm vẫn được nhận và workflow tiếp tục vào sáng hôm sau.

### Suy giảm sau 35 phút

METR quan sát thấy mọi nhóm agent được đo đều giảm độ tin cậy sau ~35 phút hoạt động liên tục. Khi thời lượng tác vụ tăng gấp đôi, tỷ lệ thất bại tăng khoảng bốn lần. Thực thi bền vững không khắc phục điều này; nó cho phép chạy lâu hơn mức độ tin cậy có thể đáp ứng. Cách an toàn là kết hợp tính bền vững với checkpoint yêu cầu HITL (con người tham gia) mới khi chạy tiếp, cùng cơ chế ngắt theo ngân sách (Bài 13) giới hạn tổng lượng tính toán bất kể thời gian thực đã trôi qua.

### Khi không nên dùng thực thi bền vững

- Lượt chạy dưới vài phút, không cần người dùng. Chi phí vận hành > lợi ích.
- Truy xuất thông tin chỉ đọc.
- Tác vụ cần hoàn tất trọn vẹn trong một cửa sổ ngữ cảnh để bảo đảm tính đúng đắn (một số tác vụ suy luận; một số tác vụ tạo nội dung một lần).

```figure
memory-consolidation
```

## Áp dụng

`code/main.py` triển khai một engine thực thi bền vững tối giản bằng thư viện chuẩn Python. Nó hỗ trợ:

- Decorator `@activity` ghi đầu vào và đầu ra vào nhật ký sự kiện JSON.
- Một hàm workflow sắp xếp thứ tự activity.
- Hàm `run_or_replay(workflow, event_log)` phát lại activity đã hoàn tất mà không thực thi lại.

Chương trình mô phỏng workflow gồm ba activity, gặp sự cố giữa chừng, rồi cho thấy (a) cách thử lại đơn giản thực thi lại mọi thứ, so với (b) phát lại chỉ chạy activity còn thiếu.

## Đóng gói

`outputs/skill-durable-execution-review.md` đánh giá một đề xuất triển khai agent chạy dài hạn về cấu trúc thực thi bền vững: activity, tính tất định, backend lưu checkpoint, trạng thái chờ người dùng và chính sách HITL khi tiếp tục.

## Bài tập

1. Chạy `code/main.py`. Quan sát chênh lệch số lần thực thi activity giữa thử lại đơn giản và phát lại. Đổi điểm xảy ra sự cố rồi cho thấy số activity được phát lại thay đổi tương ứng.

2. Sửa engine mẫu để dùng `thread_id` tường minh. Mô phỏng hai phiên đồng thời dùng chung engine và xác nhận nhật ký sự kiện của chúng không trùng nhau.

3. Chọn một activity trong engine mẫu. Thêm yếu tố không tất định (dấu thời gian thực trong một quyết định của workflow). Minh họa sự sai khác khi phát lại. Giải thích cách engine thực tế xử lý việc này (đăng ký tác động phụ, API `Workflow.now()`).

4. Đọc bài “Runtime behind production deep agents” của LangChain. Liệt kê mọi trạng thái runtime lưu lại và nêu kiểu lỗi mà mỗi trạng thái giúp xử lý.

5. Thiết kế chính sách checkpoint cho tác vụ lập trình tự chủ kéo dài 6 giờ. Bạn lưu checkpoint ở đâu? Việc tiếp tục sau sự cố diễn ra thế nào? Việc gì cần HITL mới?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Workflow | “Kịch bản của agent” | Mã điều phối tất định; có thể phát lại từ nhật ký sự kiện |
| Activity | “Một bước” | Đơn vị không tất định (gọi LLM, gọi công cụ); được ghi trước và sau khi chạy |
| Event log | “Nơi lưu trữ” | Bản ghi bền vững của mọi lần chuyển trạng thái |
| Replay | “Tiếp tục” | Chạy lại workflow; activity đã hoàn tất trả về kết quả đã ghi mà không thực thi lại |
| Checkpoint | “Điểm lưu” | Trạng thái đã lưu theo thread_id; khi tiếp tục, dùng bản mới nhất |
| thread_id | “Khóa phiên” | Định danh phân tách trạng thái bền vững theo phiên |
| 35-minute degradation | “Độ tin cậy suy giảm” | METR: tỷ lệ thành công giảm gần theo bình phương độ dài tác vụ |
| Non-determinism | “Sai khác khi phát lại” | Đồng hồ thực, giá trị ngẫu nhiên, đầu ra LLM; phải được đăng ký là tác động phụ |

## Đọc thêm

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — ngân sách, số lượt và quy tắc tiếp tục.
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cấu trúc RequestInfoEvent.
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — yêu cầu cụ thể đối với runtime.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — cấu trúc activity cho lệnh gọi LLM.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — nguồn về hiện tượng suy giảm sau 35 phút.

---

[← Bài trước: Tác tử trình duyệt và tác vụ web dài hạn](../../11-browser-agents/docs/vi.md) · [Bài sau: Ngân sách hành động, giới hạn vòng lặp và kiểm soát chi phí →](../../13-cost-governors/docs/vi.md)
