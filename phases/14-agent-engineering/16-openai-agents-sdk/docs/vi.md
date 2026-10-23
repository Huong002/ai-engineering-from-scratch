# OpenAI Agents SDK: Handoff, Guardrail, Tracing

> OpenAI Agents SDK là framework đa agent nhẹ, xây trên Responses API. Năm primitive: Agent, Handoff, Guardrail, Session, Tracing. Handoff là tool mang tên `transfer_to_<agent>`. Guardrail chặn ở đầu vào hoặc đầu ra. Tracing bật sẵn mặc định.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 06 (sử dụng công cụ)
**Time:** ~75 phút

## Mục tiêu học tập

- Kể tên năm primitive của OpenAI Agents SDK.
- Giải thích handoff: vì sao chúng được mô hình hoá thành tool, model thấy tên có dạng gì, và ngữ cảnh chuyển đi thế nào.
- Phân biệt guardrail đầu vào, guardrail đầu ra và guardrail tool; giải thích `run_in_parallel` so với chế độ blocking.
- Cài đặt một runtime thuần stdlib với handoff + guardrail + tracing kiểu span.

## Vấn đề

Agent không chuyển việc gọn gàng sẽ nhồi hết mọi thứ vào một prompt. Agent không có guardrail thì làm rò rỉ PII, xuất nội dung vi phạm chính sách, hoặc lặp vô hạn. SDK của OpenAI hệ thống hoá ba primitive khiến việc đa agent trở nên khả thi.

## Khái niệm

### Năm primitive

1. **Agent.** LLM + instruction + tool + handoff.
2. **Handoff (bàn giao).** Chuyển việc cho một agent khác. Với model, nó hiện ra như một tool tên `transfer_to_<agent_name>`.
3. **Guardrail (hàng rào bảo vệ).** Kiểm tra hợp lệ trên đầu vào (chỉ agent đầu), đầu ra (chỉ agent cuối), hoặc lúc gọi tool (mỗi function tool).
4. **Session.** Tự động lưu lịch sử hội thoại qua các lượt.
5. **Tracing (theo vết).** Span dựng sẵn cho các lần sinh LLM, gọi tool, handoff, guardrail.

### Handoff như một tool

Model thấy `transfer_to_billing_agent` trong danh sách tool của nó. Gọi tool đó là tín hiệu để runtime:

1. Sao chép ngữ cảnh hội thoại (hoặc nén lại bằng beta `nest_handoff_history`).
2. Khởi tạo agent đích với instruction của nó.
3. Tiếp tục lượt chạy với agent đích.

Đây là supervisor pattern (Bài 13 / Bài 28) được sản phẩm hoá.

### Guardrail

Ba loại:

- **Guardrail đầu vào.** Chạy trên đầu vào của agent đầu tiên. Từ chối yêu cầu không an toàn hoặc ngoài phạm vi trước khi có bất kỳ lệnh gọi LLM nào.
- **Guardrail đầu ra.** Chạy trên đầu ra của agent cuối. Bắt rò rỉ PII, vi phạm chính sách, phản hồi sai định dạng.
- **Guardrail tool.** Chạy theo từng function tool. Kiểm tra tham số, xem quyền, ghi vết thực thi.

Chế độ:

- **Parallel** (mặc định). LLM của guardrail chạy song song với LLM chính. Đuôi độ trễ thấp hơn. Nếu bị trip, phần việc của LLM chính bị bỏ (phí token).
- **Blocking** (`run_in_parallel=False`). LLM của guardrail chạy trước. Nếu bị trip, không phí token cho lệnh gọi chính.

Tripwire ném ra `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Tracing

Bật mặc định. Mỗi lần sinh LLM, gọi tool, handoff và guardrail đều phát một span. `OPENAI_AGENTS_DISABLE_TRACING=1` để tắt. `add_trace_processor(processor)` đẩy span sang backend riêng của bạn song song với của OpenAI.

### Sessions

`Session` lưu lịch sử hội thoại trong một backend (SQLite, Redis, tự viết). `Runner.run(agent, input, session=session)` (Runner là bộ chạy) tự nạp lịch sử và nối thêm.

### Chỗ mẫu này hay hỏng

- **Handoff trôi dạt.** Agent A bàn giao cho Agent B, B lại bàn giao ngược về A. Thêm bộ đếm số bước nhảy.
- **Lách guardrail.** Guardrail tool chỉ kích hoạt với function tool; tool dựng sẵn (đọc file, fetch web) cần chính sách riêng.
- **Theo vết quá đà.** Nội dung nhạy cảm nằm trong span. Ghép với quy tắc ghi nội dung GenAI của OTel (Bài 23) — lưu ở ngoài, tham chiếu bằng ID.

```figure
ae-agent-handoff
```

## Tự tay làm

`code/main.py` cài đặt hình dạng của SDK bằng stdlib:

- `Agent`, `FunctionTool`, `Handoff` (như một function tool mang ngữ nghĩa chuyển việc).
- `Runner` với guardrail đầu vào/đầu ra/tool, điều phối handoff, và bộ đếm số bước nhảy.
- Một bộ phát span đơn giản để thấy hình dạng của trace.
- Một agent điều phối, bàn giao cho billing hoặc support tuỳ truy vấn của người dùng; guardrail trip ở một đầu vào.

Chạy thử:

```
python3 code/main.py
```

Trace cho thấy hai lần handoff thành công, một lần guardrail đầu vào trip, và một cây span giống những gì SDK thật phát ra.

## Áp dụng

- **OpenAI Agents SDK** (bộ công cụ phát triển agent) cho sản phẩm ưu tiên OpenAI.
- **Claude Agent SDK** (Bài 17) cho sản phẩm ưu tiên Claude.
- **LangGraph** (Bài 13) khi bạn muốn state tường minh và resume bền vững.
- **Tự viết** khi bạn cần kiểm soát tuyệt đối (giọng nói, đa nhà cung cấp, triển khai liên kết).

## Đóng gói

`outputs/skill-agents-sdk-scaffold.md` dựng bộ khung ứng dụng Agents SDK gồm agent điều phối, handoff, guardrail đầu vào/đầu ra/tool, kho session, và một trace processor.

## Bài tập

1. Thêm bộ đếm số bước nhảy của handoff: từ chối sau N lần chuyển. Theo vết hành vi đó.
2. Cài đặt `nest_handoff_history` như một tuỳ chọn — nén các tin nhắn trước thành một bản tóm tắt trước khi chuyển.
3. Viết một guardrail đầu ra kiểu blocking. So sánh độ trễ trên các prompt sẽ bị trip so với các prompt đi qua.
4. Nối `add_trace_processor` vào một JSON logger. Mỗi span nó phát ra hình dạng gì?
5. Đọc tài liệu SDK. Chuyển món đồ chơi stdlib của bạn sang `openai-agents-python`. Bạn đã mô hình hoá sai chỗ nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Agent | "LLM + instruction" | Kiểu Agent trong SDK; sở hữu tool và handoff |
| Handoff | "Chuyển việc" | Tool mà model gọi để chuyển việc cho agent khác |
| Guardrail | "Kiểm tra chính sách" | Kiểm tra hợp lệ trên đầu vào / đầu ra / lúc gọi tool |
| Tripwire | "Guardrail bị trip" | Exception ném ra khi guardrail từ chối |
| Session | "Kho lịch sử" | Bộ nhớ hội thoại được lưu giữa các lần chạy |
| Tracing | "Span" | Khả năng quan sát dựng sẵn trên LLM + tool + handoff + guardrail |
| Blocking guardrail | "Kiểm tra tuần tự" | Guardrail chạy trước; không phí token khi bị trip |
| Parallel guardrail | "Kiểm tra đồng thời" | Guardrail chạy song song; độ trễ thấp hơn, phí token khi bị trip |

## Đọc thêm

- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — primitive, handoff, guardrail, tracing
- [Tổng quan Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — bản tương ứng kiểu Claude
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — khi nào mới cần tới handoff
- [Quy ước ngữ nghĩa GenAI của OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — chuẩn mà span của Agents SDK ánh xạ tới
