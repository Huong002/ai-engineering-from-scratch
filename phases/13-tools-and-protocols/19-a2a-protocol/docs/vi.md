# A2A — Giao thức agent với agent

> MCP là agent-với-công-cụ. A2A (Agent2Agent) là agent-to-agent (agent với agent, A2A) — một giao thức mở cho phép các agent kín xây trên những framework khác nhau phối hợp với nhau. Google công bố tháng 4/2025, hiến tặng cho Linux Foundation tháng 6/2025, đạt v1.0 tháng 4/2026 với hơn 150 tổ chức ủng hộ gồm AWS, Cisco, Microsoft, Salesforce, SAP và ServiceNow. Nó tiếp nhận ACP của IBM và bổ sung phần mở rộng thanh toán AP2. Bài này đi qua Agent Card (thẻ agent), vòng đời task và ba binding của giao thức, dùng đúng tên trên wire của A2A 1.0.1.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Giai đoạn 13, Bài 06 (kiến thức cơ bản về MCP), Giai đoạn 13, Bài 08 (máy khách MCP)
**Time:** ~75 phút

## Mục tiêu học tập

- Phân biệt trường hợp dùng agent-với-công-cụ (MCP) và agent với agent (A2A).
- Công bố một Agent Card tại `/.well-known/agent-card.json` kèm metadata skills và `supportedInterfaces`.
- Đi qua vòng đời task: `TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, `TASK_STATE_INPUT_REQUIRED`, và các trạng thái kết thúc `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_CANCELED`, `TASK_STATE_REJECTED`.
- Dùng Message mà mỗi Part chứa một trong `text`, `raw`, `url` hoặc `data`, và Artifact làm đầu ra.

## Vấn đề

Một agent chăm sóc khách hàng cần delegation (uỷ thác) việc viết báo cáo cho một agent viết chuyên dụng. Các lựa chọn trước khi có A2A:

- REST API tự làm. Chạy được nhưng mỗi cặp tích hợp là một lần làm riêng.
- Dùng chung codebase. Buộc hai agent phải chạy cùng một framework.
- MCP. Không hợp: MCP dùng để gọi công cụ, không phải để hai agent phối hợp mà vẫn giữ nguyên phần suy luận nội bộ kín của mỗi agent.

A2A lấp khoảng trống đó. Nó mô hình hoá tương tác thành việc một agent gửi task (tác vụ) cho agent khác, kèm vòng đời, message và artifact. Trạng thái nội bộ của agent được gọi vẫn kín — bên gọi chỉ thấy các chuyển trạng thái của task và những đầu ra cuối cùng.

A2A là giao thức "để các agent ở các framework khác nhau nói chuyện với nhau". Nó không thay thế MCP; hai giao thức bổ trợ cho nhau.

## Khái niệm

### Agent Card

Mọi agent tuân thủ A2A đều công bố một card tại `/.well-known/agent-card.json`:

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "version": "1.2.0",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true},
  "securitySchemes": {
    "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
  },
  "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ]
}
```

Khám phá năng lực (capability discovery) dựa trên URL: tải card về, chọn mục `supportedInterfaces` đầu tiên có `protocolBinding` mà client của bạn nói được, rồi liệt kê các skill. Input mode và output mode là media type.

### Agent Card có chữ ký

Card có thể mang một mảng `signatures`. Mỗi mục là một JWS (RFC 7515) tính trên JSON chuẩn tắc RFC 8785 của card, với trường `signatures` được bỏ ra ngoài. Bên tiêu thụ chuẩn tắc hoá card theo đúng cách đó rồi xác minh. Việc này chặn giả mạo danh tính.

### Vòng đời task

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

Client khởi tạo bằng `SendMessage`, và server tạo ra task. Agent được gọi chuyển qua các trạng thái; client thăm dò bằng `GetTask` hoặc dùng streaming (truyền phát) qua SSE với `SendStreamingMessage` và `SubscribeToTask`. Một stream mang các sự kiện `statusUpdate` và `artifactUpdate`, rồi đóng lại khi task tới trạng thái kết thúc. Không có cờ `final`.

### Message và Part

Một message có `messageId`, một `role` (`ROLE_USER` hoặc `ROLE_AGENT`), và một hoặc nhiều Part. Mỗi Part chứa đúng một trường nội dung, và tên trường đó chính là kiểu. Không có trường `kind`.

- `text`: nội dung thuần.
- `raw`: byte của file, dạng base64 trong JSON, thường kèm `filename` và `mediaType`.
- `url`: một đường dẫn tới nội dung file.
- `data`: payload JSON có cấu trúc (đầu vào có cấu trúc cho agent được gọi).

Ví dụ:

```json
{
  "messageId": "msg-001",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### Artifact

Đầu ra là Artifact, không phải chuỗi thô. Một Artifact là đầu ra có tên và có kiểu:

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

Artifact có thể được truyền theo từng chunk. Mỗi sự kiện `artifactUpdate` mang artifact kèm `append` và `lastChunk`. Bên gọi tự tích luỹ lại.

### Ba binding của giao thức

1. **JSON-RPC 2.0 qua HTTP** (`JSONRPC`). POST cho request, SSE cho streaming. Tên method viết PascalCase: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, `CreateTaskPushNotificationConfig`, `GetTaskPushNotificationConfig`, `ListTaskPushNotificationConfigs`, `DeleteTaskPushNotificationConfig`, và `GetExtendedAgentCard`.
2. **gRPC** (`GRPC`). Cho môi trường doanh nghiệp nơi gRPC là lựa chọn gốc. Cùng tên method.
3. **HTTP+JSON/REST** (`HTTP+JSON`). URL tài nguyên như `POST /message:send` và `GET /tasks/{id}`.

Cả ba binding mang cùng một mô hình dữ liệu. Mỗi mục `supportedInterfaces` nêu một binding và `protocolVersion` của nó. Client gửi header `A2A-Version: 1.0` trong mọi request, vì server đọc request thiếu header này là phiên bản 0.3.

```http
POST /a2a HTTP/1.1
Host: research.example.com
Content-Type: application/json
A2A-Version: 1.0

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{"text": "Summarize this paper."}]
    }
  }
}
```

### Giữ tính kín

Một nguyên tắc thiết kế then chốt: trạng thái nội bộ của agent được gọi là kín. Bên gọi chỉ thấy trạng thái task và artifact. Chuỗi suy nghĩ của agent được gọi, các lệnh gọi công cụ của nó, việc nó uỷ thác cho sub-agent — tất cả đều vô hình. Điều này khác MCP, nơi các lệnh gọi công cụ là minh bạch.

Lý do: A2A cho phép các đối thủ phối hợp mà không lộ nội bộ. A2A có thể chỉ là "gọi agent chăm sóc khách hàng này" mà bên gọi không cần biết agent đó hiện thực dịch vụ ra sao.

### Dòng thời gian

- **2025-04-09.** Google công bố A2A.
- **2025-06-23.** Hiến tặng cho Linux Foundation.
- **2025-08.** Tiếp nhận ACP của IBM.
- **2025-09.** Phần mở rộng AP2 (Agent Payments) ra mắt.
- **2026-04.** Phát hành v1.0 với hơn 150 tổ chức ủng hộ.

### Quan hệ với MCP

| Khía cạnh | MCP | A2A |
|-----------|-----|-----|
| Trường hợp dùng | Agent với công cụ | Agent với agent |
| Tính kín | Lệnh gọi công cụ minh bạch | Suy luận nội bộ kín |
| Bên gọi điển hình | Runtime của agent | Một agent khác |
| Trạng thái | Kết quả gọi công cụ | Task có vòng đời |
| Phân quyền | OAuth 2.1 (Giai đoạn 13 · 16) | Agent Card `securitySchemes` + `securityRequirements` |
| Truyền tải | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

Dùng MCP khi bạn muốn gọi một công cụ cụ thể. Dùng A2A khi bạn muốn uỷ thác cả một task cho agent khác. Nhiều hệ thống production dùng cả hai: một agent dùng MCP cho tầng công cụ và A2A cho tầng phối hợp.

```figure
a2a-task-lifecycle
```

## Áp dụng

`code/main.py` hiện thực một harness A2A tối giản: agent viết công bố card của nó, agent nghiên cứu gửi cho nó một request `SendMessage` với một Part PDF và một chỉ dẫn dạng text, và task đi qua `TASK_STATE_WORKING` → `TASK_STATE_INPUT_REQUIRED` → `TASK_STATE_WORKING` → `TASK_STATE_COMPLETED` rồi trả về một artifact dạng text. Toàn bộ dùng stdlib; dùng transport trong bộ nhớ để tập trung vào hình dạng của message.

Nên chú ý những gì:

- Hình dạng JSON của Agent Card.
- Việc gán task id phía server và các chuyển trạng thái.
- Part được định kiểu theo trường nội dung nào đang có mặt.
- Nhánh `TASK_STATE_INPUT_REQUIRED` giữa task.
- Artifact trả về khi hoàn thành.

## Đóng gói

Bài này tạo ra `outputs/skill-a2a-agent-spec.md`. Với một agent mới cần được agent khác gọi tới, skill này tạo ra JSON Agent Card, schema skills và bản thiết kế endpoint.

## Bài tập

1. Chạy `code/main.py`. Lần theo toàn bộ vòng đời task, gồm cả chỗ dừng `TASK_STATE_INPUT_REQUIRED` nơi agent được gọi hỏi lại cho rõ.

2. Thêm một Agent Card có chữ ký. Đặt một mục JWS trong `signatures` với `alg` là `HS256`, ký trên JSON chuẩn tắc của card khi đã bỏ trường `signatures`. Viết một bộ xác minh và xác nhận nó thất bại với card đã bị sửa.

3. Hiện thực streaming cho task bằng `SendStreamingMessage`: agent viết phát ra `task`, ba chunk `artifactUpdate`, và một `statusUpdate` với `TASK_STATE_COMPLETED`, rồi đóng stream. Bên gọi tích luỹ các chunk.

4. Thiết kế một agent A2A bọc một MCP server. Ánh xạ mỗi công cụ MCP thành một skill A2A. Ghi lại các đánh đổi — mất đi tính kín nào?

5. Đọc thông báo A2A v1.0 và tìm ra tính năng duy nhất mà tính tới tháng 4/2026 chưa framework nào hiện thực. (Gợi ý: nó liên quan tới việc uỷ thác task qua nhiều chặng.)

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Giao thức mở cho phép các agent kín phối hợp với nhau |
| Agent Card | "`/.well-known/agent-card.json`" | Metadata được công bố, mô tả skills và `supportedInterfaces` của một agent |
| Skill | "Một đơn vị gọi được" | Một thao tác có tên mà agent hỗ trợ (tương tự công cụ MCP) |
| Task | "Đơn vị uỷ thác" | Một hạng mục công việc có vòng đời và artifact cuối cùng |
| Message | "Đầu vào của task" | Mang các Part (`text`, `raw`, `url`, `data`) |
| Part | "Mẩu có kiểu" | Đúng một trong `text` / `raw` / `url` / `data`, kèm `mediaType` tuỳ chọn; không có trường `kind` |
| Artifact | "Đầu ra của task" | Đầu ra có tên và có kiểu, trả về khi hoàn thành |
| AP2 | "Agent Payments Protocol" | Phần mở rộng thanh toán xây trên A2A; ký card là phần lõi của A2A (`signatures`) |
| Opacity | "Phối hợp dạng hộp đen" | Nội bộ của agent được gọi bị ẩn khỏi bên gọi |
| `TASK_STATE_INPUT_REQUIRED` | "Task tạm dừng" | Trạng thái bị ngắt khi agent cần thêm thông tin |

## Đọc thêm

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — đặc tả chuẩn của A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — hiện thực tham chiếu và các SDK
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1): `docs/specification.md` ở tag này và `specification/a2a.proto` quy chuẩn mà bài học bám theo
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — chuyển giao quản trị tháng 6/2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — lộ trình và đà phát triển của các đối tác
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — ghi chú phát hành v1.0 và hướng dẫn tương thích ngược

---

[← Bài trước: Xác thực MCP trong production: đăng ký gắn issuer và token gắn issuer](../../18-mcp-auth-production/docs/vi.md) · [Bài sau: OpenTelemetry GenAI →](../../20-opentelemetry-genai/docs/vi.md)
