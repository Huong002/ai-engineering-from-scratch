# Mô hình actor (actor model) cho agent (tác tử) — Thông điệp bất đồng bộ và runtime có kiểu

> Agent dưới dạng actor: trao đổi thông điệp bất đồng bộ, handler hướng sự kiện, cô lập lỗi, tương tranh tự nhiên. AutoGen v0.4 (Microsoft Research, tháng 1 năm 2025) đã thiết kế lại việc điều phối agent quanh mô hình này; framework nay đã chuyển sang chế độ bảo trì, với Microsoft Agent Framework (bản xem trước công khai tháng 10 năm 2025) là kế nhiệm cho môi trường production.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 12 (mẫu workflow)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả mô hình actor: agent là actor, thông điệp là IPC duy nhất, cô lập lỗi theo từng actor.
- Kể tên ba lớp API của AutoGen v0.4 — Core, AgentChat, Extensions — và mục đích của từng lớp.
- Giải thích vì sao tách việc gửi thông điệp khỏi việc xử lý lại mang lại cô lập lỗi và tương tranh tự nhiên.
- Tự viết một runtime actor bằng stdlib Python và chuyển luồng review code hai agent lên đó.

## Vấn đề

Hầu hết framework agent đều đồng bộ: một agent tạo ra, một agent tiêu thụ, trong cùng một call stack. Lỗi làm sập cả stack. Tương tranh bị gắn thêm bên ngoài. Phân tán đòi hỏi viết lại.

Câu trả lời của AutoGen v0.4: mô hình actor. Mỗi agent là một actor có hộp thư riêng. Thông điệp là cách tương tác duy nhất. Runtime tách việc gửi khỏi việc xử lý. Lỗi bị cô lập trong một actor. Tương tranh có sẵn. Phân tán chỉ là đổi transport.

## Khái niệm

### Actor

Một actor có:

- Trạng thái riêng (không bao giờ bị bên ngoài chạm trực tiếp).
- Một hộp thư vào (hàng đợi thông điệp).
- Một handler: `receive(message) -> effects` trong đó effects có thể là "trả lời", "gửi cho actor khác", "sinh actor mới", "cập nhật trạng thái", "tự dừng".

Hai actor không thể chia sẻ bộ nhớ. Chúng chỉ có thể gửi thông điệp cho nhau.

### Ba lớp API

AutoGen v0.4 chia bề mặt của nó thành ba phần:

1. **Core.** Framework actor cấp thấp. `AgentRuntime`, `Agent`, `Message`, `Topic`. Trao đổi thông điệp bất đồng bộ, hướng sự kiện.
2. **AgentChat.** API cấp cao theo hướng tác vụ (thay cho ConversableAgent của v0.2). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Các tích hợp — OpenAI, Anthropic, Azure, tools, memory.

### Vì sao việc tách rời quan trọng

Trong mô hình v0.2, gọi `agent_a.chat(agent_b)` sẽ chặn agent_a cho tới khi agent_b trả về. Ở v0.4, `send(agent_b, msg)` đặt thông điệp vào hộp thư của agent_b rồi trả về ngay. Runtime gửi sau đó. Ba hệ quả:

- **Cô lập lỗi.** Agent B sập không làm Agent A sập — runtime bắt lỗi trong handler của B và quyết định làm gì (ghi log, thử lại, dead-letter).
- **Tương tranh tự nhiên.** Nhiều thông điệp cùng bay một lúc; các actor xử lý hộp thư của mình song song.
- **Sẵn sàng phân tán.** Hộp thư + transport là cùng một abstraction dù actor nằm trong cùng tiến trình hay trên máy khác.

### Các topology

- **RoundRobinGroupChat.** Các agent đến lượt (turn) theo vòng xoay cố định.
- **SelectorGroupChat.** Một agent chọn lọc quyết định ai đi tiếp dựa trên ngữ cảnh hội thoại.
- **Magentic-One.** Đội đa agent tham chiếu cho duyệt web, chạy code, xử lý tệp. Xây trên AgentChat.

### Khả năng quan sát

Hỗ trợ OpenTelemetry được tích hợp sẵn. Mỗi thông điệp phát ra một span; các lệnh gọi tool mang thuộc tính `gen_ai.*` theo quy ước ngữ nghĩa OTel GenAI 2026 (Bài 23).

### Trạng thái: chế độ bảo trì

Đầu năm 2026: AutoGen v0.7.x ổn định cho nghiên cứu và tạo mẫu. Microsoft đã chuyển phát triển tích cực sang Microsoft Agent Framework, kế nhiệm cho production (bản xem trước công khai 1 tháng 10 năm 2025; bản GA 1.0 nhắm tới cuối Q1 2026). Các mẫu AutoGen chuyển sang rất gọn — mô hình actor là ý tưởng bền vững.

```figure
actor-mailbox
```

## Tự tay làm

`code/main.py` triển khai một runtime actor bằng stdlib:

- `Message` — payload có kiểu với `sender`, `recipient`, `topic`, `body`.
- `Actor` — lớp trừu tượng với `receive(message, runtime)`.
- `Runtime` — event loop với một hàng đợi chia sẻ, việc gửi thông điệp, cô lập lỗi.
- Một demo hai actor: `ReviewerAgent` review code, `ChecklistAgent` chạy một checklist; chúng trao đổi thông điệp cho tới khi đồng thuận.

Chạy thử:

```
python3 code/main.py
```

Vết chạy cho thấy việc gửi thông điệp, một lỗi mô phỏng ở một actor mà không làm sập actor kia, và sự hội tụ về một phán quyết chung.

## Áp dụng

- **AutoGen v0.4/v0.7** (bảo trì) — ổn định cho nghiên cứu, tạo mẫu, các mẫu đa agent.
- **Microsoft Agent Framework** — kế nhiệm cho production (bản xem trước công khai tháng 10 năm 2025); cùng những ý tưởng mô hình actor trong một API được làm mới.
- **Topology swarm của LangGraph** (Bài 13) — mẫu tương tự qua bàn giao bằng tool chung.
- **Runtime actor tự viết** — khi bạn cần một transport cụ thể (NATS, RabbitMQ, gRPC).

## Đóng gói

`outputs/skill-actor-runtime.md` sinh ra một runtime actor tối giản cộng một template đội (RoundRobin hoặc Selector) cho một tác vụ đa agent cho trước.

## Bài tập

1. Thêm hàng đợi dead-letter: khi handler ném lỗi, giữ thông điệp lỗi lại để con người kiểm tra. Trong đồ chơi của bạn, DLQ bị chạm tới bao nhiêu lần?
2. Cài đặt `SelectorGroupChat`: một actor chọn lọc quyết định ai xử lý thông điệp tiếp theo dựa trên trạng thái hội thoại.
3. Thêm transport phân tán: đổi hàng đợi trong tiến trình thành một server JSON-over-HTTP để các actor chạy ở các tiến trình riêng.
4. Nối một span OTel cho mỗi thông điệp (hoặc một stand-in không làm gì). Phát `gen_ai.agent.name`, `gen_ai.operation.name` theo Bài 23.
5. Đọc bài viết kiến trúc AutoGen v0.4. Chuyển đồ chơi của bạn sang API `autogen_core` thật. Bạn đã bỏ qua điều gì mà quan trọng trong production?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Actor | "Agent" | Trạng thái riêng + hộp thư + handler; không chia sẻ bộ nhớ |
| Message | "Event" | Payload có kiểu; cách duy nhất để các actor tương tác |
| Inbox | "Mailbox" | Hàng đợi thông điệp đang chờ riêng của từng actor |
| Runtime | "Agent host" | Event loop định tuyến thông điệp và cô lập lỗi |
| Topic | "Channel" | Tuyến publish-subscribe có tên giữa các actor |
| Fault isolation | "Let it crash" | Một actor hỏng không làm hỏng các actor khác |
| RoundRobinGroupChat | "Fixed-rotation team" | Các agent lần lượt theo thứ tự |
| SelectorGroupChat | "Context-routed team" | Actor chọn lọc quyết định ai đi tiếp |
| Magentic-One | "Reference team" | Đội đa agent cho web + code + tệp |

## Đọc thêm

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — bài viết về lần thiết kế lại
- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — lựa chọn thay thế dạng đồ thị
- [Quy ước ngữ nghĩa GenAI của OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — các span AutoGen phát ra mặc định
