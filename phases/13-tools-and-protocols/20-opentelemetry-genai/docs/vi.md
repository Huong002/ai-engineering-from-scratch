# OpenTelemetry GenAI — Truy vết lời gọi công cụ từ đầu đến cuối

> Một agent gọi năm công cụ, ba máy chủ MCP và hai sub-agent. Bạn cần một trace (vết) duy nhất xuyên suốt tất cả. Các quy ước ngữ nghĩa OpenTelemetry GenAI — với attribute (thuộc tính) ổn định từ v1.37 trở lên — là chuẩn của năm 2026, được Datadog, Langfuse, Arize Phoenix, OpenLLMetry và AgentOps hỗ trợ sẵn. Bài học này nêu tên các thuộc tính bắt buộc, đi qua cây phân cấp span (agent → LLM → công cụ), và đóng gói một span emitter thuần stdlib mà bạn có thể cắm vào bất kỳ exporter OTel nào.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Giai đoạn 13, Bài 07 (máy chủ MCP), Giai đoạn 13, Bài 08 (máy khách MCP)
**Time:** ~75 phút

## Mục tiêu học tập

- Nêu tên các thuộc tính OTel GenAI bắt buộc cho một span LLM và một span thực thi công cụ.
- Xây dựng một cây phân cấp trace bao trùm vòng lặp agent, lời gọi LLM, lời gọi công cụ và việc điều phối của máy khách MCP.
- Quyết định nội dung nào cần ghi lại (tự chọn bật) và nội dung nào cần che (mặc định).
- Phát span tới một collector cục bộ (Jaeger, Langfuse) mà không phải viết lại mã công cụ.

## Vấn đề

Một lần gỡ lỗi tháng 2 năm 2026: người dùng báo "agent của tôi có lúc mất 30 giây để trả lời, lúc khác chỉ 3 giây." Không có trace nào. Log có ghi lời gọi LLM, nhưng không ghi việc điều phối công cụ, không ghi vòng round-trip tới máy chủ MCP, không ghi sub-agent. Bạn chỉ còn cách đoán. Cuối cùng bạn phát hiện: một máy chủ MCP thỉnh thoảng bị treo lúc khởi động nguội.

Không có truy vết đầu-cuối, bạn không thể tìm ra lỗi này. OTel GenAI giải quyết được.

Các quy ước được chốt trong giai đoạn 2025-2026 trong nhóm OpenTelemetry semantic-conventions. Chúng định nghĩa tên thuộc tính ổn định để Datadog, Langfuse, Phoenix, OpenLLMetry và AgentOps đều phân tích cùng một span theo cùng một cách. Chỉ cần instrumentation (đo đạc) một lần; gửi được tới bất kỳ backend nào.

## Khái niệm

### Cây phân cấp span

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Toàn bộ nằm lồng dưới một trace id. Các span id liên kết quan hệ cha-con.

### Các thuộc tính bắt buộc

Theo semconv 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — chuỗi model được yêu cầu (ví dụ `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — model thực sự phục vụ.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — id phản hồi của nhà cung cấp để đối chiếu.

Với span công cụ:

- `gen_ai.tool.name` — định danh công cụ.
- `gen_ai.tool.call.id` — id của lời gọi cụ thể.
- `gen_ai.tool.description` — mô tả công cụ (tuỳ chọn).

Với span agent:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Các loại span

- `SpanKind.CLIENT` cho các lời gọi vượt ranh giới tiến trình (nhà cung cấp LLM, máy chủ MCP).
- `SpanKind.INTERNAL` cho các bước trong vòng lặp của chính agent và việc thực thi công cụ.

### Ghi nội dung theo kiểu tự chọn bật

Mặc định, span chỉ mang số liệu và thời gian — không mang prompt hay completion. Payload lớn và PII đều tắt mặc định. Đặt `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và các biến môi trường ghi nội dung cụ thể để bao gồm nội dung. Hãy xem xét kỹ trước khi bật trên prod.

### Sự kiện trên span

Có thể thêm các sự kiện theo token thành sự kiện của span:

- `gen_ai.content.prompt` — các message đầu vào.
- `gen_ai.content.completion` — các message đầu ra.
- `gen_ai.content.tool_call` — lời gọi công cụ như đã ghi nhận.

Các sự kiện được sắp theo thời gian trong một span để phát lại chi tiết.

### Exporter

Span OTel xuất được tới:

- **Jaeger / Tempo.** Mã nguồn mở, chạy tại chỗ.
- **Langfuse.** Chuyên cho observability (khả năng quan sát) LLM; trực quan hoá mức dùng token.
- **Arize Phoenix.** Kết hợp đánh giá và truy vết.
- **Datadog.** Thương mại; phân tích sẵn các thuộc tính `gen_ai.*`.
- **Honeycomb.** Hướng cột; thân thiện với truy vấn.

Tất cả đều nói OTLP, định dạng truyền tải. Mã của bạn không cần quan tâm.

### Truyền ngữ cảnh qua MCP

Khi máy khách MCP gọi máy chủ, hãy chèn header traceparent của W3C vào request. Streamable HTTP hỗ trợ header chuẩn. Stdio không mang header HTTP một cách tự nhiên; lộ trình 2026 của spec có bàn tới việc thêm trường `_meta.traceparent` vào các lời gọi JSON-RPC.

Cho tới khi điều đó được phát hành: hãy tự đưa traceparent vào `_meta` của mọi request. Máy chủ ghi lại trace id.

### Số liệu

Song song với span, semconv GenAI định nghĩa các số liệu:

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

Dùng những cái này cho dashboard không cần chi tiết từng lời gọi.

### Lớp AgentOps

AgentOps (thành lập năm 2024) chuyên về observability cho GenAI. Nó bọc các framework phổ biến (LangGraph, Pydantic AI, CrewAI) để tự động phát span OTel. Hữu ích nếu stack của bạn dùng framework được hỗ trợ; nếu không thì dùng instrumentation thủ công.

```figure
t3-span-waterfall
```

## Áp dụng

`code/main.py` phát ra các span có hình dạng OTel tới stdout (theo dạng gần giống OTLP-JSON) cho một agent gọi LLM, điều phối hai công cụ và thực hiện một round-trip MCP. Không có exporter thật — bài học tập trung vào hình dạng span và bộ thuộc tính. Dán output vào một trình xem tương thích OTLP hoặc chỉ cần đọc nó.

Cần chú ý:

- Trace id được chia sẻ cho mọi span.
- Liên kết cha-con được mã hoá qua `parentSpanId`.
- Các thuộc tính `gen_ai.*` bắt buộc đã được điền.
- Ghi nội dung mặc định tắt; một kịch bản bật nó qua biến môi trường.

## Đóng gói

Bài học này tạo ra `outputs/skill-otel-genai-instrumentation.md`. Cho một codebase agent, skill tạo ra một kế hoạch instrumentation: thêm span ở đâu, điền thuộc tính nào và nhắm tới exporter nào.

## Bài tập

1. Chạy `code/main.py`. Đếm số span và xác định cái nào là CLIENT, cái nào là INTERNAL.

2. Bật ghi nội dung (biến môi trường) và xác nhận các sự kiện `gen_ai.content.prompt` và `gen_ai.content.completion` xuất hiện. Lưu ý hệ quả với PII.

3. Thêm số liệu thực thi công cụ `gen_ai.tool.execution.duration` và phát nó như một mẫu histogram cho mỗi lời gọi.

4. Truyền một traceparent từ span agent cha vào trường `_meta.traceparent` của một request MCP. Kiểm tra rằng máy chủ MCP sẽ thấy cùng trace id.

5. Đọc spec semconv OTel GenAI. Tìm một thuộc tính được nêu trong semconv mà mã của bài này KHÔNG phát ra. Thêm nó.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Chuẩn mở cho trace, số liệu, log |
| GenAI semconv | "GenAI semantic conventions" | Tên thuộc tính ổn định cho span LLM / công cụ / agent |
| `gen_ai.*` | "Không gian tên thuộc tính" | Mọi thuộc tính GenAI dùng chung tiền tố này |
| Span | "Thao tác có thời lượng" | Một đơn vị công việc có bắt đầu, kết thúc và thuộc tính |
| Trace | "Quan hệ tổ tiên xuyên span" | Cây các span chia sẻ cùng trace id |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Gợi ý về hướng của span |
| OTLP | "OpenTelemetry Line Protocol" | Định dạng truyền tải cho exporter |
| Opt-in content | "Ghi prompt / completion" | Mặc định tắt; bật bằng biến môi trường |
| traceparent | "Header W3C" | Truyền ngữ cảnh trace qua các dịch vụ |
| Exporter | "Bộ gửi riêng cho backend" | Thành phần gửi span tới Jaeger / Datadog / v.v. |

## Đọc thêm

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — quy ước chuẩn cho span, số liệu và sự kiện GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — danh sách thuộc tính của span LLM và span thực thi công cụ
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — span `invoke_agent` ở cấp agent
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — nguồn tham chiếu chuẩn trên GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — hướng dẫn tích hợp cho môi trường production

---

[← Bài trước: A2A](../../19-a2a-protocol/docs/vi.md) · [Bài sau: Lớp routing (định tuyến) LLM →](../../21-llm-routing-layer/docs/vi.md)
