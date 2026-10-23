# Runtime production: Hàng đợi, Sự kiện, Cron

> Agent production chạy trên sáu dạng runtime (môi trường chạy): request-response, streaming, durable execution (thực thi bền vững), queue-based background, event-driven, và scheduled. Chọn dạng trước khi chọn framework. Observability (khả năng quan sát) là yếu tố chịu lực ở mọi dạng.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 13 (LangGraph), Giai đoạn 14, Bài 22 (giọng nói)
**Time:** ~60 phút

## Mục tiêu học tập

- Kể tên sáu dạng runtime production và ghép mỗi dạng với một mẫu framework / sản phẩm.
- Giải thích vì sao durable execution (LangGraph) quan trọng với các tác vụ dài hạn.
- Mô tả runtime hướng sự kiện và khi nào Claude Managed Agents phù hợp.
- Giải thích luận điểm observability là yếu tố chịu lực với agent nhiều bước.

## Vấn đề

Agent production thất bại theo những cách mà một notebook Jupyter không hề lộ ra: timeout mạng ở bước 37, người dùng cúp máy giữa cuộc gọi thoại, cron job chết khi máy khởi động lại, worker (tiến trình thực thi) nền hết bộ nhớ. Dạng runtime quyết định lỗi nào còn sống sót được.

## Khái niệm

### Request-response

- HTTP đồng bộ. Người dùng chờ đến khi xong.
- Chỉ khả thi với tác vụ ngắn (<30s).
- Stack: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Observability: access log HTTP chuẩn + span OTel.

### Streaming

- SSE hoặc WebSocket để xuất dần kết quả.
- LiveKit mở rộng điều này sang WebRTC cho thoại/video (Bài 22).
- Stack: framework nào hỗ trợ streaming + frontend xử lý được SSE/WS.
- Observability: thời gian từng chunk, độ trễ token đầu tiên, độ trễ đuôi.

### Durable execution

- Trạng thái được checkpoint (điểm lưu trạng thái) sau mỗi bước; tự động chạy tiếp khi lỗi.
- Mô hình actor của AutoGen v0.4 cô lập lỗi vào một agent (Bài 14).
- Điểm khác biệt cốt lõi của LangGraph (Bài 13).
- Thiết yếu khi số bước không biết trước và chi phí phục hồi cao.

### Queue-based / background

- Job vào hàng đợi, worker nhận việc, kết quả trả về qua webhook hoặc pub/sub.
- Thiết yếu cho agent dài hạn (hàng chục đến hàng trăm bước mỗi tác vụ, theo thông báo computer use của Anthropic).
- Stack: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), hoặc tự viết.
- Observability: độ sâu hàng đợi, phân bố độ trễ từng job, kích thước DLQ.

### Event-driven

- Agent đăng ký theo trigger: email mới, PR được mở, cron kích hoạt.
- Claude Managed Agents phủ sẵn trường hợp này (Bài 17).
- CrewAI Flows (Bài 15) cấu trúc hoá workflow tất định hướng sự kiện.
- Observability: nguồn trigger, độ trễ từ sự kiện đến lúc bắt đầu, độ trễ của agent.

### Scheduled

- Agent dạng cron chạy định kỳ.
- Kết hợp với durable execution để lượt chạy đêm bị lỗi sẽ chạy tiếp ở nhịp sau.
- Stack: Kubernetes CronJob + framework bền vững; dịch vụ host sẵn (Render cron, Vercel cron).

### Mẫu triển khai 2026

- **CrewAI Flows** cho production hướng sự kiện.
- **Agno** FastAPI không trạng thái cho microservice Python.
- **Mastra** server adapter (Express, Hono, Fastify, Koa) để nhúng.
- **Pipecat Cloud / LiveKit Cloud** cho thoại được quản lý (Bài 22).
- **Claude Managed Agents** cho async dài hạn được host sẵn.

### Observability là yếu tố chịu lực

Không có span OpenTelemetry GenAI (Bài 23) cùng một backend Langfuse/Phoenix/Opik (Bài 24), bạn không thể debug một agent nhiều bước đã lỗi ở bước 40. Điều này không phải tuỳ chọn khi lên production. Đó là khác biệt giữa "chúng ta debug nhanh" và "chúng ta chạy lại từ đầu với nhiều log hơn".

### Nơi runtime production thất bại

- **Chọn sai dạng.** Chọn request-response cho tác vụ 5 phút. Người dùng bỏ đi; worker chất đống; retry cộng dồn.
- **Không có DLQ.** Worker hàng đợi không có dead-letter. Job lỗi biến mất.
- **Việc nền không nhìn thấy được.** Agent nền chạy mà không xuất trace. Lỗi vô hình cho tới khi người dùng báo.
- **Bỏ qua trạng thái bền.** Mọi lượt chạy > 30 giây mà bạn không thể cho phép khởi động lại đều cần durable execution.

```figure
wb-runtime-shapes
```

## Tự tay làm

`code/main.py` là bản demo nhiều dạng bằng stdlib:

- Endpoint request-response (hàm thường).
- Handler streaming (generator).
- Worker hàng đợi có DLQ.
- Registry trigger sự kiện.
- Bộ lập lịch dạng cron.

Chạy thử:

```bash
python3 code/main.py
```

Kết quả: năm trace thể hiện hành vi của từng dạng trên cùng một tác vụ. Cùng logic agent, khác lớp vỏ ngoài. Durable execution (dạng thứ sáu) được cố ý trình bày ở Bài 13 với checkpointing của LangGraph.

## Áp dụng

- **Request-response** cho UX kiểu chat.
- **Streaming** cho phản hồi dần dần.
- **Durable** cho tác vụ dài hạn.
- **Queue** cho batch / async / chạy lâu.
- **Event** cho khả năng phản ứng của agent.
- **Cron** cho việc dọn dẹp (củng cố bộ nhớ, eval, báo cáo chi phí).

## Đóng gói

`outputs/skill-runtime-shape.md` chọn một dạng runtime cho tác vụ và nối các yêu cầu observability.

## Bài tập

1. Chuyển vòng lặp ReAct của Bài 01 sang cả sáu dạng trong stack của bạn. Dạng nào hợp với bề mặt sản phẩm nào?
2. Thêm DLQ vào demo hàng đợi. Giả lập 10% job lỗi; hiển thị kích thước DLQ.
3. Viết một agent eval kích hoạt bằng cron, chạy mỗi đêm trên 20 trace hàng đầu trong ngày của bạn.
4. Cài streaming có backpressure: nếu client chậm, tạm dừng agent. Điều này tương tác thế nào với ngân sách lượt?
5. Đọc tài liệu Claude Managed Agents. Khi nào bạn nên chuyển một agent dài hạn tự host sang dạng được quản lý?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Request-response | "Đồng bộ" | Người dùng chờ; chỉ tác vụ ngắn |
| Streaming | "SSE / WS" | Xuất dần kết quả; UX tốt hơn; đo được độ trễ từng chunk |
| Durable execution | "Chạy tiếp khi lỗi" | Trạng thái đã checkpoint; khởi động lại từ bước cuối |
| Queue-based | "Job nền" | Producer / pool worker / DLQ |
| Event-driven | "Dựa trên trigger" | Agent phản ứng với sự kiện bên ngoài |
| DLQ | "Dead-letter queue" | Chỗ đỗ cho các job lỗi |
| Claude Managed Agents | "Harness được host" | Async dài hạn do Anthropic host, có cache + compaction |

## Đọc thêm

- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — chi tiết về durable execution
- [Tổng quan Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — async dài hạn được host sẵn
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "hàng chục đến hàng trăm bước mỗi tác vụ"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — cô lập lỗi theo mô hình actor

---

[← Bài trước: Các mẫu điều phối (orchestration): Supervisor, Swarm, Hierarchical](../../28-orchestration-patterns/docs/vi.md) · [Bài sau: Phát triển agent dựa trên đánh giá →](../../30-eval-driven-agent-development/docs/vi.md)
