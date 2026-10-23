# Đánh đổi (trade-off) giữa các framework (khung) agent — Điều phối theo đồ thị, vai trò và Actor

> Mọi framework đều bán cùng một bản demo (agent nghiên cứu dựng báo cáo) và giấu cùng một lỗi (schema trạng thái đánh nhau với lớp điều phối (orchestration)). Hãy chọn framework có trừu tượng (abstraction) khớp với hình dạng bài toán của bạn; phần còn lại chỉ là keo dán bạn phải viết hai lần.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 11, Bài 09 (Function calling), Giai đoạn 11, Bài 16 (LangGraph)
**Time:** ~45 phút

## Vấn đề

Bạn có một tác vụ cần nhiều hơn một lần gọi LLM. Có thể đó là một quy trình nghiên cứu (lập kế hoạch, tìm kiếm, tóm tắt, trích dẫn). Có thể đó là một pipeline review code (phân tích diff, phê bình, vá, kiểm chứng). Có thể đó là một trợ lý nhiều lượt đặt vé máy bay, viết email và lập báo cáo chi phí. Bạn chọn một framework.

Ba ngày sau, bạn phát hiện trừu tượng của framework bị rò. CrewAI cho bạn vai trò nhưng lại chống lại bạn khi "researcher" cần trao một kế hoạch có cấu trúc cho "writer". AutoGen cho bạn trò chuyện giữa các agent nhưng không có trạng thái hạng nhất, nên checkpoint của bạn chỉ là một pickle của log hội thoại. LangGraph cho bạn đồ thị trạng thái nhưng buộc bạn đặt tên mọi chuyển tiếp trước khi biết agent sẽ làm gì. Agno cho bạn trừu tượng một agent, thứ gào lên khi bạn thử toả ra ba worker chạy song song.

Cách sửa không phải là "chọn framework tốt nhất". Mà là khớp trừu tượng cốt lõi của framework với hình dạng bài toán của bạn. Bài học này vẽ ra tấm bản đồ đó.

## Khái niệm

![Ma trận framework agent: trừu tượng cốt lõi so với hình dạng bài toán](../assets/framework-matrix.svg)

Bốn framework thống trị bối cảnh 2026. Trừu tượng cốt lõi của chúng không giống nhau.

| Framework | Trừu tượng cốt lõi | Phù hợp nhất | Không phù hợp nhất |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — trạng thái có kiểu, node, cạnh có điều kiện, checkpointer. | Quy trình có trạng thái tường minh và điểm ngắt chờ con người; agent production cần gỡ lỗi kiểu time-travel. | Brainstorming lỏng, dẫn dắt theo vai trò, nơi cấu trúc liên kết chưa biết. |
| **CrewAI** | `Crew` — vai trò (mục tiêu, lai lịch), task, process (tuần tự hoặc phân cấp). | Quy trình nhập vai hoặc dẫn dắt theo persona với kế hoạch tuyến tính/phân cấp ngắn. | Bất cứ thứ gì cần trạng thái vượt quá lịch sử lượt của crew; phân nhánh phức tạp. |
| **AutoGen** | Cặp `ConversableAgent` — hai hoặc nhiều agent nói theo lượt cho tới khi gặp điều kiện thoát. | *Đối thoại* đa agent (người dạy-học trò, người đề xuất-phê bình, người làm-người duyệt) nơi tư duy nảy ra từ cuộc trò chuyện. | Quy trình tất định với DAG đã biết; bất cứ thứ gì cần trạng thái bền qua lần khởi động lại. |
| **Agno** | `Agent` — một LLM + tool + bộ nhớ, có thể ghép thành team. | Agent đơn và team nhẹ, xây nhanh; đa phương thức mạnh và có sẵn driver lưu trữ. | Đồ thị phân nhánh sâu, tường minh với reducer tuỳ biến. |

### "Trừu tượng" thực sự nghĩa là gì

Trừu tượng cốt lõi của một framework là thứ bạn vẽ lên bảng trắng khi trình bày kiến trúc.

- **LangGraph** → bạn vẽ một đồ thị. Node là các bước, cạnh là chuyển tiếp, và đối tượng trạng thái tại mọi điểm đều có kiểu. Mô hình trong đầu là một máy trạng thái.
- **CrewAI** → bạn vẽ một sơ đồ tổ chức. Mỗi vai trò có một bản mô tả công việc và một người quản lý định tuyến task. Mô hình trong đầu là một nhóm nhỏ chuyên gia.
- **AutoGen** → bạn vẽ một tin nhắn Slack riêng. Hai agent nhắn cho nhau; agent thứ ba tham gia nếu bạn cần người điều hành. Mô hình trong đầu là trò chuyện.
- **Agno** → bạn vẽ một cái hộp duy nhất với các tool treo quanh nó. Đặt các hộp cạnh nhau để thành một team. Mô hình trong đầu là "agent kèm pin sẵn".

### Câu hỏi về trạng thái

Trạng thái là nơi hầu hết lựa chọn framework sụp đổ trong production.

- **LangGraph.** Trạng thái có kiểu (`TypedDict` hoặc model Pydantic), reducer theo từng trường, checkpointer hạng nhất (SQLite/Postgres/Redis). Resume, interrupt và time-travel miễn phí. *(Xem Giai đoạn 11 · 16.)*
- **CrewAI.** Trạng thái chảy dưới dạng chuỗi giữa các task qua trường `context`, hoặc có cấu trúc qua `output_pydantic`. Không có kho bền theo từng crew ngay từ đầu; bạn tự gắn thêm nếu crew phải sống sót qua lần khởi động lại.
- **AutoGen.** Trạng thái là lịch sử trò chuyện và `context` do người dùng định nghĩa. Bản ghi hội thoại được lưu; trạng thái quy trình tuỳ ý thì không, trừ khi bạn viết adapter.
- **Agno.** Driver lưu trữ có sẵn (SQLite, Postgres, Mongo, Redis, DynamoDB) gắn vào một `Agent` qua `storage=` — phiên trò chuyện và bộ nhớ người dùng tự động được lưu. Đây không phải checkpointer đồ thị đầy đủ; đó là kho phiên.

### Câu hỏi về phân nhánh

Mọi agent không tầm thường đều phân nhánh. Ai quyết định nhánh mới là điều quan trọng.

- **LangGraph** — bạn quyết định, qua các cạnh có điều kiện. Định tuyến là một hàm Python với các nhánh có tên. Nhánh là hạng nhất trong đồ thị đã biên dịch; checkpointer ghi lại nhánh nào đã được đi.
- **CrewAI** — người quản lý quyết định ở chế độ phân cấp; ở chế độ tuần tự bạn quyết định lúc xây dựng. Định tuyến nằm ngầm trong danh sách task; không có "if" hạng nhất nào ngoài prompt của người quản lý.
- **AutoGen** — các agent quyết định qua trò chuyện. Phân nhánh nảy sinh từ việc ai nói tiếp theo. `GroupChatManager` chọn người nói kế tiếp; bạn có thể tự viết một `speaker_selection_method` nhưng mặc định là do LLM dẫn dắt.
- **Agno** — agent quyết định bằng việc gọi tool nào tiếp theo. Team có chế độ coordinator/router/collaborator; phân nhánh vượt ra ngoài đó là trách nhiệm của lập trình viên.

### Câu hỏi về khả năng quan sát

- **LangGraph** — OpenTelemetry qua LangSmith hoặc bất kỳ exporter OTel nào. Mỗi chuyển tiếp node là một span trace; checkpoint đồng thời là trace có thể phát lại. LangSmith là lựa chọn của chính hãng; Langfuse/Phoenix cũng có adapter.
- **CrewAI** — OpenTelemetry hạng nhất từ cuối 2025; tích hợp với Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** — tích hợp OpenTelemetry qua `autogen-core`; AgentOps và Opik có connector. Độ chi tiết của tracing là theo từng tin nhắn agent, không phải theo node.
- **Agno** — cờ `monitoring=True` có sẵn cùng các exporter OpenTelemetry; tích hợp chặt với Langfuse cho trace theo phiên.

### Chi phí và độ trễ

Cả bốn framework đều thêm overhead cho mỗi lần gọi (logic framework, kiểm chứng, tuần tự hoá). Thứ tự tăng dần tương đối của overhead: Agno ≈ LangGraph < CrewAI ≈ AutoGen. Khác biệt bị chi phối bởi việc framework định tuyến bằng LLM nhiều tới đâu. Người quản lý phân cấp của CrewAI tiêu token để quyết định ai đi tiếp; `GroupChatManager` của AutoGen cũng vậy. LangGraph chỉ tiêu token ở nơi bạn viết `llm.invoke`. Đường một agent của Agno rất mỏng.

Khi chi phí mỗi lần chạy quan trọng, hãy ưu tiên định tuyến tường minh (cạnh của LangGraph, `speaker_selection_method` của AutoGen) hơn định tuyến do LLM chọn.

### Khả năng tương tác

- **LangGraph** ↔ tool, retriever, LLM của **LangChain**. Adapter MCP hạng nhất (tool được nhập vào như các MCP server).
- **CrewAI** ↔ tool kế thừa từ `BaseTool`; tool của LangChain, LlamaIndex và MCP đều thích nghi được. Uỷ thác crew-với-crew qua `allow_delegation=True`.
- **AutoGen** → `FunctionTool` bọc bất kỳ callable Python nào; có adapter MCP. Gắn chặt vào hệ sinh thái AG2 cho các mẫu agent-với-agent, một dạng lock-in (bị khoá vào nhà cung cấp).
- **Agno** → decorator `@tool` hoặc lớp con của BaseTool; có adapter MCP; tool có thể chia sẻ giữa các agent và team.

## Kỹ năng

> Bạn có thể giải thích, trong một câu, vì sao một framework cụ thể là đúng cho một bài toán agent cụ thể.

Danh sách kiểm tra trước khi xây:

1. **Vẽ hình dạng.** Đây là đồ thị (trạng thái có kiểu, chuyển tiếp có tên)? Là nhập vai (các chuyên gia bàn giao việc)? Là trò chuyện (agent nói tới khi xong)? Hay một agent đơn với tool?
2. **Quyết định ai phân nhánh.** Lập trình viên quyết định phân nhánh → LangGraph. Agent-quản-lý quyết định → CrewAI phân cấp. Trò chuyện nảy sinh → AutoGen. Quyết định bằng lời gọi tool → Agno.
3. **Kiểm tra ngân sách trạng thái.** Bạn có cần resume-từ-checkpoint? Time-travel? Con người ngắt giữa chừng? Nếu có, LangGraph là mặc định; phiên của Agno phủ được trạng thái trong phạm vi hội thoại.
4. **Kiểm tra ngân sách chi phí.** Định tuyến do LLM chọn tốn thêm token mỗi lượt. Nếu agent chạy hàng nghìn lần một ngày, hãy ưu tiên định tuyến tường minh.
5. **Tính ngân sách cho overhead của framework.** Mỗi framework là thêm một dependency (phụ thuộc). Nếu tác vụ chỉ là hai lần gọi LLM và một tool, hãy viết 30 dòng Python thuần; không framework nào rẻ hơn không dùng framework.

Đừng với tay lấy framework trước khi bạn vẽ được đồ thị, sơ đồ tổ chức, cuộc trò chuyện hoặc cái hộp agent. Đừng chọn framework buộc bạn phải đánh nhau với mô hình trạng thái của nó vì đúng thứ bạn thực sự cần.

## Ma trận quyết định

| Hình dạng bài toán | Framework ưu tiên | Vì sao |
|---------------|---------------------|-----|
| DAG quy trình với trạng thái có kiểu, con người phê duyệt, chạy dài | LangGraph | Trạng thái hạng nhất, checkpointer, interrupt, time-travel. |
| Pipeline nghiên cứu / viết với các vai trò khác biệt | CrewAI (tuần tự) hoặc subgraph LangGraph | Mỗi task một vai trò thì rẻ để diễn đạt trong CrewAI; nâng lên LangGraph khi phân nhánh phức tạp. |
| Đối thoại người đề xuất-phê bình hoặc người dạy-học trò | AutoGen | Trò chuyện hai agent chính là hình dạng gốc của nó. |
| Agent đơn với tool, phiên, bộ nhớ | Agno | Cách dựng mỏng nhất, có sẵn lưu trữ và bộ nhớ. |
| Hàng nghìn nhánh toả song song với reducer | LangGraph + `Send` | Là cái duy nhất có API phát tán song song hạng nhất. |
| Nguyên mẫu nhanh, chưa muốn cam kết framework | Python thuần + SDK của nhà cung cấp | Không framework chính là framework nhanh nhất. |

```figure
l5-framework-fit
```

## Bài tập

1. **Dễ.** Lấy cùng một tác vụ — "nghiên cứu trụ sở của Anthropic, viết bản tóm tắt 200 từ, trích dẫn nguồn" — và hiện thực nó trong LangGraph (bốn node: plan, search, write, cite) và trong CrewAI (ba vai trò: researcher, writer, editor). Báo cáo chi phí token mỗi lần chạy và số dòng code.
2. **Trung bình.** Dựng cùng tác vụ đó trong AutoGen (researcher ↔ writer trò chuyện, editor tham gia qua `GroupChat`) và Agno (một agent đơn với `search_tools` và `write_tools`, cộng một kho phiên). Xếp hạng bốn cách hiện thực theo (a) chi phí mỗi lần chạy, (b) khả năng resume sau khi crash, (c) khả năng chèn bước con người phê duyệt trước bước write.
3. **Khó.** Viết script cây quyết định `pick_framework.py` nhận một mô tả bài toán ngắn (JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) và trả về một khuyến nghị kèm lý do một câu. Kiểm chứng nó trên sáu trường hợp do bạn tự thiết kế.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Orchestration | "Cách các agent phối hợp" | Lớp quyết định node/vai trò/agent nào chạy tiếp theo. |
| Durable state | "Resume sau khi khởi động lại" | Trạng thái sống sót qua cái chết của tiến trình, gắn với checkpoint hoặc kho phiên. |
| LLM-selected routing | "Để model quyết định" | Một LLM lập kế hoạch chọn bước tiếp theo mỗi lượt; linh hoạt nhưng tốn token ở mọi quyết định. |
| Explicit routing | "Lập trình viên quyết định" | Một hàm Python hoặc cạnh tĩnh chọn bước tiếp theo; rẻ và kiểm toán được. |
| Crew | "Một team của CrewAI" | Vai trò + task + process (tuần tự hoặc phân cấp) gói thành một thứ chạy được. |
| GroupChat | "Trò chuyện đa agent của AutoGen" | Một hội thoại được quản lý giữa N agent với bộ chọn người nói. |
| Team (Agno) | "Agno đa agent" | Chế độ route / coordinate / collaborate trên một tập agent. |
| StateGraph | "Đồ thị của LangGraph" | Trừu tượng gồm trạng thái có kiểu, node, cạnh có điều kiện, checkpointer. |

## Đọc thêm

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointer, interrupt, time-travel.
- [CrewAI documentation](https://docs.crewai.com/) — Crew, Flow, Agent, Task, Process.
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, team, tool.
- [Agno documentation](https://docs.agno.com/) — Agent, Team, Workflow, lưu trữ, bộ nhớ.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — thư viện mẫu (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) không phụ thuộc framework.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — agent loop (vòng lặp agent) mà mọi framework khoác áo lên.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — bài báo thiết kế của AutoGen.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — nền tảng nhập vai mà các chồng persona kiểu CrewAI xây trên đó.
- Giai đoạn 11 · 16 (LangGraph) — framework mà bài học này lấy làm mốc so sánh.
- Giai đoạn 11 · 19 (Reflexion) — một mẫu khớp gọn với LangGraph nhưng gượng gạo với CrewAI.
- Giai đoạn 11 · 22 (Khả năng quan sát trong production) — cách gắn đo đạc cho framework bạn chọn.

---

[← Bài trước: Máy trạng thái (state machine) của agent](../../16-langgraph-state-machines/docs/vi.md)
