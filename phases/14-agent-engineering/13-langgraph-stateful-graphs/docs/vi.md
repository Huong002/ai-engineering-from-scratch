# Điều phối Stateful Graph (đồ thị có trạng thái) — Thực thi bền bỉ và checkpoint

> Agent là một máy trạng thái; node (nút) là hàm; edge (cạnh) là chuyển tiếp; state được checkpoint (điểm lưu) sau mỗi node. Tiếp tục từ bất kỳ lỗi nào, ngay tại checkpoint thành công gần nhất. LangGraph là tham chiếu năm 2026 cho mô hình điều phối trạng thái tầng thấp này.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 12 (mẫu workflow)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả mô hình lõi của LangGraph: máy trạng thái với state có kiểu, node là hàm, edge có điều kiện, và checkpoint sau mỗi node.
- Kể tên bốn năng lực mà tài liệu nhấn mạnh: thực thi bền bỉ, streaming, human-in-the-loop, bộ nhớ toàn diện.
- Giải thích ba topology điều phối mà LangGraph hỗ trợ: supervisor, peer-to-peer (swarm), phân cấp (subgraph lồng nhau).
- Cài đặt một state graph bằng stdlib với state có kiểu, edge có điều kiện, và chu trình checkpoint/resume.

## Vấn đề

Agent và workflow gặp chung một vấn đề: khi một lượt chạy 40 bước hỏng ở bước 38, bạn muốn chạy tiếp từ bước 38 chứ không phải làm lại từ đầu. Những mô hình state hạng hai buộc người vận hành phải chắp vá cơ chế thử lại quanh một thư viện vốn giả định mỗi lần chạy đều là mới.

Câu trả lời thiết kế của LangGraph: state là một đối tượng có kiểu hạng nhất, mọi thay đổi đều tường minh, và checkpoint được lưu sau mỗi node. Resume chỉ là một lời gọi `load_state(session_id)`.

## Khái niệm

### Đồ thị

Một đồ thị được định nghĩa bởi:

- **State type.** Một dict có kiểu (hoặc Pydantic model) mà mọi node đều đọc và thay đổi.
- **Nodes.** Hàm thuần `(state) -> state_update`. Các cập nhật được gộp vào state sau khi hàm trả về.
- **Edges.** Chuyển tiếp có điều kiện hoặc trực tiếp giữa các node.
- **Entry and exit.** Hai node lính canh `START` và `END` đánh dấu ranh giới.

Ví dụ: một agent có các node `classify`, `refund`, `bug`, `sales`, `done` — một workflow định tuyến được diễn đạt thành đồ thị.

### Thực thi bền bỉ

Sau khi mỗi node trả về, runtime tuần tự hoá state và ghi vào một checkpointer (SQLite, Postgres, Redis, hoặc tự viết). Khi hỏng ở bước N, runtime có thể gọi `resume(session_id)` và chạy tiếp từ bước N+1 với state chính xác.

Tài liệu LangGraph nêu rõ những người dùng production mà điều này thực sự quan trọng: Klarna, Uber, J.P. Morgan. Điểm đáng nói không phải hình dạng đồ thị; mà là hình dạng đồ thị cộng với checkpoint khiến việc phục hồi trở nên rẻ.

### Streaming

Mỗi node có thể phát ra output từng phần. Đồ thị stream các sự kiện delta theo từng node về cho bên gọi, để UI cập nhật ngay trong lúc đồ thị chạy.

### Human-in-the-loop

Xem và sửa state giữa các node. Cách làm: tạm dừng trước một node quan trọng, đưa state ra cho con người xem, nhận các thay đổi, rồi chạy tiếp. Checkpointer làm việc này dễ vì state đã được tuần tự hoá sẵn.

### Bộ nhớ

Ngắn hạn (trong một lượt chạy — lịch sử hội thoại nằm trong state) và dài hạn (qua nhiều lượt chạy — bền vững nhờ checkpointer cộng thêm một kho dài hạn riêng). LangGraph tích hợp với các hệ bộ nhớ bên ngoài (Mem0, tự viết) thông qua tool.

### Ba topology

1. **Supervisor.** Một LLM định tuyến trung tâm điều phối tới các subagent chuyên trách. `create_supervisor()` trong `langgraph-supervisor` (dù năm 2026 đội LangChain khuyến nghị làm việc này trực tiếp qua tool call để kiểm soát context tốt hơn).
2. **Swarm / peer-to-peer.** Các agent bàn giao trực tiếp qua một bề mặt tool dùng chung. Không có bộ định tuyến trung tâm.
3. **Phân cấp.** Các supervisor quản lý sub-supervisor, hiện thực bằng những subgraph lồng nhau.

### Chỗ mẫu này hỏng

- **Checkpoint quá nhỏ.** Chỉ checkpoint các lượt hội thoại sẽ khiến state của tool và các lần ghi bộ nhớ không thể phục hồi. Toàn bộ state phải được tuần tự hoá.
- **Node không tất định.** Resume giả định cùng đầu vào node sẽ tạo ra cùng cập nhật state. Seed ngẫu nhiên, đồng hồ thực, API bên ngoài đều phải được ghi lại.
- **Lạm dụng edge có điều kiện.** Một đồ thị mà mọi edge đều có điều kiện là một máy trạng thái không thể suy luận nổi. Hãy ưu tiên chuỗi tuyến tính với vài nhánh thỉnh thoảng.

```figure
langgraph-state
```

## Tự tay làm

`code/main.py` cài đặt một state graph bằng stdlib:

- `State` — một dict có kiểu gồm `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — callable nhận state và trả về một dict cập nhật.
- `StateGraph` — node + edge + edge có điều kiện + run + resume.
- `SQLiteCheckpointer` (bản giả lập trong bộ nhớ) — tuần tự hoá state sau mỗi node; `load(session_id)` khôi phục lại.
- Một đồ thị demo: classify -> branch(refund / bug / sales) -> cổng con người -> send.

Chạy thử:

```
python3 code/main.py
```

Trace sẽ cho thấy lần chạy đầu hỏng ở cổng con người, state được lưu lại, rồi resume tạo ra output cuối cùng.

## Áp dụng

- **LangGraph** — bản tham chiếu, sẵn sàng cho production. Dùng `create_react_agent`, `create_supervisor`, hoặc tự dựng đồ thị của bạn.
- **AutoGen v0.4** (Bài 14) — lựa chọn theo actor model cho các tình huống cần độ đồng thời cao.
- **Claude Agent SDK** (Bài 17) — harness được quản lý với session store dựng sẵn.
- **Tự viết** — khi bạn cần kiểm soát chính xác hình dạng state hoặc backend của checkpointer.

## Đóng gói

`outputs/skill-state-graph.md` sinh ra một state graph theo hình dạng LangGraph trong bất kỳ runtime đích nào, với checkpoint và resume được nối sẵn.

## Bài tập

1. Thêm một edge có điều kiện từ `classify` sang `end` khi độ tin cậy phân loại dưới ngưỡng. Resume lượt chạy sau khi con người đặt `route` thủ công.
2. Thay bản giả lập kiểu SQLite bằng một checkpointer SQLite thật. Đo chi phí tuần tự hoá trên mỗi bước.
3. Cài đặt edge song song: hai node chạy đồng thời, gộp bằng một reducer (hàm gộp trạng thái) tuỳ chỉnh. State bất biến mang lại gì ở đây?
4. Đọc tài liệu tham chiếu `langgraph-supervisor`. Chuyển bản đồ chơi sang `create_supervisor`. So sánh hình dạng trace.
5. Thêm streaming: mỗi node phát ra state từng phần trong khi chạy. In các delta ra ngay khi chúng tới.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| State graph | "Agent như một máy trạng thái" | State có kiểu + node + edge + reducer |
| Checkpointer | "Backend lưu trữ" | Tuần tự hoá state sau mỗi node; cho phép resume |
| Reducer | "Bộ gộp state" | Hàm kết hợp state hiện tại với cập nhật của một node |
| Conditional edge | "Nhánh" | Edge được chọn bởi một hàm của state |
| Subgraph | "Đồ thị lồng nhau" | Một đồ thị được dùng như node bên trong đồ thị khác |
| Durable execution | "Chạy tiếp sau lỗi" | Khởi động lại ở node thành công gần nhất với state chính xác |
| Supervisor | "LLM định tuyến" | Bộ điều phối trung tâm cho các subagent chuyên trách |
| Swarm | "Agent P2P" | Các agent bàn giao qua tool dùng chung; không có bộ định tuyến trung tâm |

## Đọc thêm

- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — tài liệu tham chiếu
- [Tài liệu tham chiếu langgraph-supervisor](https://reference.langchain.com/python/langgraph/supervisor/) — API của mẫu supervisor
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — lựa chọn theo actor model
- [Tổng quan Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — session store và subagent

---

[← Bài trước: Các mẫu workflow của Anthropic: Đơn giản hơn là phức tạp](../../12-anthropic-workflow-patterns/docs/vi.md) · [Bài sau: Mô hình actor (actor model) cho agent (tác tử) →](../../14-autogen-actor-model/docs/vi.md)
