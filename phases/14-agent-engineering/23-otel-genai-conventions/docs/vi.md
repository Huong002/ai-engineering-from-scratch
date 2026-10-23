# Quy ước ngữ nghĩa GenAI của OpenTelemetry

> Nhóm GenAI SIG của OpenTelemetry (ra mắt tháng 4/2024) định nghĩa schema chuẩn cho telemetry của agent. Tên span (đơn vị công việc được đo), attribute (thuộc tính) và quy tắc ghi nội dung cùng theo một bộ quy ước ngữ nghĩa (semantic conventions), hội tụ giống nhau giữa các nhà cung cấp, để trace agent mang cùng một ý nghĩa trên Datadog, Grafana, Jaeger và Honeycomb.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 13 (LangGraph), Giai đoạn 14, Bài 24 (nền tảng quan sát)
**Time:** ~60 phút

## Mục tiêu học tập

- Kể tên các nhóm span GenAI: model/client, agent, tool.
- Phân biệt span `invoke_agent` kiểu CLIENT và INTERNAL, và khi nào dùng loại nào.
- Liệt kê các attribute cấp cao nhất của GenAI: tên nhà cung cấp, model trong request, ID nguồn dữ liệu.
- Giải thích hợp đồng ghi nội dung: tùy chọn bật (opt-in), `OTEL_SEMCONV_STABILITY_OPT_IN`, khuyến nghị dùng tham chiếu bên ngoài.

## Vấn đề

Mỗi nhà cung cấp tự phát minh tên span cho riêng mình. Đội vận hành rốt cuộc phải dựng dashboard riêng cho từng framework. Nhóm GenAI SIG của OpenTelemetry sửa chuyện này bằng cách định nghĩa một chuẩn duy nhất để cả hệ sinh thái hướng tới.

## Khái niệm

### Các nhóm span

1. **Span model / client.** Bao các lời gọi LLM thô. Do SDK của nhà cung cấp (Anthropic, OpenAI, Bedrock) và adapter model của framework phát ra.
2. **Span agent.** `create_agent` (khi agent được khởi tạo) và `invoke_agent` (khi nó chạy).
3. **Span tool.** Một span cho mỗi lần gọi tool; nối với span agent bằng quan hệ cha-con.

### Đặt tên span agent

- Tên span: `invoke_agent {gen_ai.agent.name}` nếu có tên; nếu không thì dùng `invoke_agent`.
- Loại span:
  - **CLIENT** — cho dịch vụ agent từ xa (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — cho framework agent chạy trong cùng tiến trình (LangChain, CrewAI, ReAct cục bộ).

### Các attribute chính

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — ID của model.
- `gen_ai.response.model` — model thực tế được dùng (có thể khác request do routing).
- `gen_ai.agent.name` — định danh agent.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — cho RAG: đã tra cứu corpus hay kho nào.

Có cả quy ước riêng theo từng công nghệ cho Anthropic, Azure AI Inference, AWS Bedrock, OpenAI.

### Ghi lại nội dung

Quy tắc mặc định: instrumentation (đo đạc) KHÔNG NÊN ghi lại input/output theo mặc định. Việc ghi là tùy chọn bật qua:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Mẫu khuyến nghị cho production: lưu nội dung ở nơi khác (S3, kho log của bạn), ghi tham chiếu lên span (ID con trỏ, không ghi nguyên văn). Đây là lớp phòng thủ chống đầu độc nội dung của Bài 27, được nối vào phần quan sát.

### Độ ổn định

Hầu hết quy ước còn ở mức thử nghiệm tính đến tháng 3/2026. Bật bản xem trước ổn định bằng:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ ánh xạ attribute GenAI gốc vào schema LLM Observability của nó. Các backend khác (Grafana, Honeycomb, Jaeger) hỗ trợ attribute thô.

### Chỗ mẫu này hay hỏng

- **Ghi cả prompt đầy đủ vào span.** PII, secret, dữ liệu khách hàng nằm trong trace mà đội vận hành đọc được. Hãy lưu ở nơi khác.
- **Thiếu `gen_ai.provider.name`.** Dashboard đa nhà cung cấp vỡ khi không rõ nguồn nào.
- **Span không có liên kết cha.** Span tool bị mồ côi. Luôn truyền context.
- **Không đặt tùy chọn ổn định.** Attribute của bạn có thể bị đổi tên khi backend nâng cấp.

```figure
ae-genai-span-tree
```

## Tự tay làm

`code/main.py` cài đặt một bộ phát span bằng stdlib khớp với quy ước GenAI:

- `Span` với schema attribute GenAI.
- `Tracer` với `start_span`, context lồng nhau.
- Một lượt chạy agent theo kịch bản phát ra: `create_agent`, `invoke_agent` (INTERNAL), span cho từng tool, span `chat` cho các lời gọi LLM.
- Một chế độ ghi nội dung lưu prompt ở nơi khác và ghi ID lên span.

Chạy thử:

```
python3 code/main.py
```

Kết quả: một cây span với đầy đủ attribute GenAI bắt buộc, và một "kho ngoài" cho thấy các tham chiếu nội dung được bật tùy chọn.

## Áp dụng

- **Datadog LLM Observability** (v1.37+) ánh xạ attribute gốc.
- **Langfuse / Phoenix / Opik** (Bài 24) — tự động đo đạc cả hệ sinh thái.
- **Jaeger / Honeycomb / Grafana Tempo** — trace OTel thô; dựng dashboard từ attribute GenAI.
- **Tự vận hành** — chạy OTel Collector với processor GenAI.

## Đóng gói

`outputs/skill-otel-genai.md` nối span OTel GenAI vào một agent đang có, kèm mặc định ghi nội dung và lưu tham chiếu bên ngoài.

## Bài tập

1. Đo đạc vòng lặp ReAct của Bài 01 bằng `invoke_agent` (INTERNAL) + span cho từng tool. Gửi tới một instance Jaeger.
2. Thêm chế độ ghi nội dung "chỉ tham chiếu": prompt vào SQLite, attribute của span chỉ mang ID dòng.
3. Đọc spec của `gen_ai.data_source.id`. Nối nó vào tìm kiếm Mem0 của Bài 09.
4. Đặt `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và kiểm tra attribute của bạn không bị collector đổi tên.
5. Dựng dashboard: "tool lỗi nào tương quan với model nào" chỉ từ attribute GenAI.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| GenAI SIG | "nhóm GenAI của OpenTelemetry" | Nhóm làm việc của OTel định nghĩa schema |
| invoke_agent | "span agent" | Tên của span đại diện một lượt chạy agent |
| CLIENT span | "lời gọi từ xa" | Span cho lời gọi tới dịch vụ agent từ xa |
| INTERNAL span | "trong tiến trình" | Span cho lượt chạy agent trong cùng tiến trình |
| gen_ai.provider.name | "nhà cung cấp" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "nguồn RAG" | Corpus/kho nào đã được truy xuất |
| Content capture | "ghi log prompt" | Ghi lại message theo tùy chọn bật; ở production hãy lưu ở nơi khác |
| Stability opt-in | "chế độ xem trước" | Biến môi trường để ghim quy ước thử nghiệm |

## Đọc thêm

- [Quy ước ngữ nghĩa GenAI của OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — span GenAI theo mặc định
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — span OTel có sẵn
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — lan truyền W3C trace context
