# Máy trạng thái (state machine) của agent — Đồ thị, nút, điểm lưu

> Một vòng lặp ReAct viết tay chỉ là `while True`. Cũng vòng lặp đó viết thành đồ thị tường minh thì bạn có thể checkpoint, interrupt, rẽ nhánh và time-travel qua nó. Agent không đổi. Chỉ có bộ khung (harness) quanh nó đổi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 11, Bài 09 (Function calling), Giai đoạn 11, Bài 14 (Model Context Protocol)
**Time:** ~75 phút

## Vấn đề

Bạn ship một agent gọi hàm. Nó chạy được ba lượt, rồi có gì đó hỏng: model gọi một tool trả về 500, người dùng đổi ý giữa task, hoặc agent quyết định hoàn tiền một đơn hàng mà không có người duyệt. Vòng `while True:` không có móc nối nào. Bạn không tạm dừng được, không tua lại được, và không rẽ nhánh sang "nếu model chọn tool kia thì sao". Ngay khi bạn ship thứ này ra ngoài demo, agent thành một hộp đen: hoặc đã chạy được, hoặc không.

Bước tiếp theo hiển nhiên khi bạn đã nhìn ra. Agent vốn đã là một máy trạng thái — system prompt cộng lịch sử message cộng các tool call đang chờ cộng hành động kế tiếp. Hãy làm máy trạng thái đó tường minh: nút cho "model suy nghĩ", "một tool chạy", "một người duyệt", và cạnh cho các chuyển tiếp có điều kiện giữa chúng. Khi đồ thị đã tường minh, bộ khung được bốn thứ miễn phí: checkpointing (lưu trạng thái giữa các bước), interrupts (tạm dừng chờ người), streaming (stream token và các sự kiện trung gian), và time-travel (tua về một trạng thái trước rồi thử nhánh khác).

Phần cài đặt tham chiếu cho abstraction này là LangGraph. Nó không phải một agent framework theo nghĩa LangChain ("đây là AgentExecutor, chúc may mắn"). Nó là một graph runtime với state hạng nhất, persistence hạng nhất và interrupt hạng nhất. Vòng lặp agent là thứ bạn vẽ ra, không phải thứ bạn viết tay.

## Khái niệm

![LangGraph StateGraph: các nút, các cạnh và checkpointer](../assets/langgraph-stategraph.svg)

Một `StateGraph` có ba thứ.

1. **Trạng thái (state).** Một dict có kiểu (TypedDict hoặc Pydantic model) chảy qua đồ thị. Mỗi nút nhận toàn bộ trạng thái và trả về một cập nhật một phần, LangGraph gộp bằng một *reducer* (bộ gộp) theo từng field — `operator.add` cho các list cần tích luỹ, mặc định là ghi đè.
2. **Nút.** Hàm Python `state -> partial_state`. Mỗi nút là một bước rời rạc: "gọi model", "chạy tools", "tóm tắt".
3. **Cạnh (edge).** Chuyển tiếp giữa các nút. Cạnh tĩnh đi một chỗ. Cạnh có điều kiện (conditional edge) nhận một hàm router `state -> next_node_name` để đồ thị rẽ nhánh theo output của model.

Bạn compile đồ thị. Compile gắn topology, gắn một checkpointer (tuỳ chọn nhưng thiết yếu cho production), và trả về một runnable. Bạn gọi nó với một trạng thái ban đầu và một `thread_id`. Mỗi bước thực thi lưu lại một checkpoint khoá theo `(thread_id, checkpoint_id)`.

### Bốn siêu năng lực

**Checkpointing.** Mỗi chuyển tiếp nút ghi trạng thái mới vào một store (in-memory cho test, Postgres/Redis/SQLite cho prod). Tiếp tục bằng cách gọi lại đồ thị với cùng `thread_id`. Đồ thị chạy tiếp từ chỗ đã dừng.

**Interrupts.** Đánh dấu một nút bằng `interrupt_before=["human_review"]` và thực thi dừng trước khi nút đó chạy. Trạng thái vẫn được lưu. API của bạn trả lời người dùng "đang chờ duyệt". Một request sau đó tới cùng `thread_id` với `Command(resume=...)` sẽ chạy tiếp.

**Streaming.** `graph.stream(state, mode="updates")` trả về các delta trạng thái ngay khi chúng xảy ra. `mode="messages"` stream token của LLM bên trong các nút model. `mode="values"` trả về ảnh chụp đầy đủ. Bạn chọn cái gì hiện lên UI.

**Time-travel.** `graph.get_state_history(thread_id)` trả về toàn bộ log checkpoint. Truyền bất kỳ `checkpoint_id` trước đó vào `graph.invoke` và bạn rẽ nhánh từ điểm đó. Rất tốt để debug ("nếu model chọn tool B thì sao?") và cho các test hồi quy replay trace production.

### Reducer là điểm mấu chốt

Mỗi field trạng thái đều có một reducer. Phần lớn giá trị mặc định là ổn — giá trị mới ghi đè giá trị cũ. Nhưng list message cần `operator.add` để message mới được nối thêm thay vì thay thế. Các cạnh song song gộp cập nhật của chúng qua reducer. Nếu hai nút cùng cập nhật `messages` mà bạn quên `Annotated[list, add_messages]`, nút thứ hai thắng một cách âm thầm và bạn mất nửa lượt. Reducer là thứ duy nhất tinh tế trong thư viện; làm đúng nó thì phần còn lại tự khớp.

### Đồ thị ReAct trong bốn nút

Một agent ReAct production là bốn nút và hai cạnh:

1. `agent` — gọi LLM với lịch sử message hiện tại. Trả về message của assistant (có thể chứa tool_calls).
2. `tools` — thực thi mọi tool_calls trong message assistant cuối cùng, nối kết quả tool vào dưới dạng message tool.
3. Một cạnh có điều kiện từ `agent` định tuyến tới `tools` nếu message cuối có tool_calls, ngược lại tới `END`.
4. Một cạnh tĩnh từ `tools` quay lại `agent`.

Chỉ có thế. Bạn được toàn bộ vòng ReAct (Thought → Action → Observation → Thought → …) với checkpointing, interrupts và streaming, trong khoảng 40 dòng code.

### StateGraph vs Send (fanout)

`Send(node_name, state)` cho phép một nút phát các đồ thị con song song. Ví dụ: agent quyết định truy vấn ba retriever cùng lúc. Mỗi `Send` sinh ra một lần thực thi song song của nút đích; output của chúng gộp qua reducer trạng thái. Đây là cách LangGraph diễn đạt pattern orchestrator-workers mà không cần primitive threading.

### Đồ thị con

Một đồ thị đã compile có thể là một nút trong đồ thị khác. Đồ thị ngoài thấy một nút duy nhất; đồ thị trong có trạng thái riêng và checkpoint riêng. Đây là cách các team xây agent supervisor-worker: đồ thị supervisor định tuyến ý định người dùng tới đồ thị con worker theo từng domain.

```figure
l5-state-graph-ledger
```

## Tự tay làm

### Bước 1: trạng thái và các nút

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` là reducer làm list message tích luỹ thay vì ghi đè. Quên nó là bug LangGraph phổ biến nhất.

### Bước 2: chạy với một thread

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Mỗi update là một dict `{node_name: state_delta}`. Frontend của bạn có thể stream những cái này lên UI để người dùng thấy "agent đang suy nghĩ… đang gọi search_web… đã có kết quả… đang trả lời".

### Bước 3: thêm interrupt human-in-the-loop

Đánh dấu một nút để thực thi tạm dừng trước khi nó chạy.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

Trạng thái, checkpoint và thread đều tồn tại qua interrupt. Không có gì nằm trong memory ngoài lúc thực thi.

### Bước 4: time-travel để debug

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Truyền `None` làm input sẽ replay từ checkpoint đã cho; truyền một giá trị sẽ nối nó vào trạng thái của checkpoint đó trước khi chạy tiếp. Đây là cách tái hiện một lượt agent hỏng mà không phải chạy lại toàn bộ hội thoại.

### Bước 5: đổi checkpointer cho production

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis và Postgres đều có sẵn. `MemorySaver` dành cho test. Bất cứ thứ gì cần tồn tại qua restart đều cần một store thật.

## Kỹ năng

> Bạn xây agent như đồ thị, không phải như vòng `while True`.

Trước khi với tới LangGraph, hãy làm một bản thiết kế 60 giây:

1. **Đặt tên các nút.** Mỗi quyết định rời rạc hoặc hành động có tác dụng phụ là một nút. "Agent suy nghĩ", "tool chạy", "reviewer duyệt", "response stream". Nếu bạn không liệt kê được, task chưa có hình dạng agent.
2. **Khai báo trạng thái.** TypedDict tối thiểu với reducer cho mọi field list. Đừng nhét tất cả vào `messages`; kéo các field đặc thù của task (một `plan` đang làm, một biến đếm `budget`, một list `retrieved_docs`) lên cấp cao nhất.
3. **Vẽ các cạnh.** Tĩnh trừ khi bước kế tiếp phụ thuộc output của model. Mọi cạnh có điều kiện cần một hàm router với các nhánh có tên.
4. **Chọn checkpointer ngay từ đầu.** `MemorySaver` cho test, Postgres/Redis/SQLite cho mọi thứ khác. Đừng ship mà không có — không checkpointer nghĩa là không resume, không interrupt, không time-travel.
5. **Quyết định interrupt trước khi tool chạy, không phải sau.** Phê duyệt đặt trên cạnh đi vào một nút có tác dụng phụ để bạn huỷ được trước khi gây hại; kiểm tra hợp lệ đặt trên cạnh đi ra khỏi model để bạn từ chối các lệnh gọi xấu một cách rẻ.
6. **Mặc định stream.** `mode="updates"` cho UI, `mode="messages"` cho stream từng token bên trong các nút model, `mode="values"` cho ảnh chụp đầy đủ khi eval.

Đừng ship một agent LangGraph không có checkpointer. Đừng ship một cái interrupt *sau* tác dụng phụ. Đừng ship một field `messages` không có `add_messages` làm reducer.

## Bài tập

1. **Dễ.** Cài đặt đồ thị ReAct bốn nút ở trên với một tool máy tính và một tool web-search. Kiểm tra rằng `list(app.get_state_history(config))` trả về ít nhất bốn checkpoint cho một hội thoại hai lượt.
2. **Trung bình.** Thêm một nút `planner` chạy trước `agent` và ghi một `plan: list[str]` có cấu trúc vào trạng thái. Cho `agent` đánh dấu các bước của plan là xong. Làm test fail nếu `plan` bị mất qua một lần resume từ checkpoint (sai reducer).
3. **Khó.** Xây một đồ thị supervisor định tuyến giữa ba đồ thị con (`researcher`, `writer`, `reviewer`) bằng `Send`. Mỗi đồ thị con có trạng thái và checkpointer riêng. Thêm `interrupt_before=["writer"]` trên đồ thị ngoài để người có thể duyệt bản tóm tắt nghiên cứu. Xác nhận rằng time-travel từ một checkpoint trước chỉ chạy lại nhánh đã rẽ.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| StateGraph | "Đồ thị LangGraph" | Đối tượng builder mà bạn thêm nút và cạnh vào trước khi compile. |
| Reducer | "Cách field gộp lại" | Một hàm `(old, new) -> merged` được áp dụng khi một nút trả về cập nhật cho field đó; mặc định là ghi đè, `add_messages` thì nối thêm. |
| Thread | "Một ID hội thoại" | Một chuỗi `thread_id` giới hạn phạm vi mọi checkpoint của một phiên. |
| Checkpoint | "Một trạng thái bị tạm dừng" | Ảnh chụp được lưu của toàn bộ trạng thái đồ thị sau một chuyển tiếp nút, khoá theo `(thread_id, checkpoint_id)`. |
| Interrupt | "Tạm dừng chờ người" | `interrupt_before` / `interrupt_after` dừng thực thi tại ranh giới một nút; chạy tiếp bằng `Command(resume=...)`. |
| Time-travel | "Rẽ nhánh từ một bước trước" | `graph.invoke(None, config_with_old_checkpoint_id)` replay từ checkpoint đó đi tiếp. |
| Send | "Phát đồ thị con song song" | Một constructor mà nút có thể trả về để sinh N lần thực thi song song của một nút đích. |
| Subgraph | "Một đồ thị đã compile làm nút" | Một StateGraph đã compile dùng làm nút trong đồ thị khác; giữ phạm vi trạng thái riêng. |

## Đọc thêm

- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — tham chiếu chuẩn cho StateGraph, reducer, checkpointer và interrupt.
- [Khái niệm LangGraph: state, reducer, checkpointer](https://langchain-ai.github.io/langgraph/concepts/low_level/) — mô hình tư duy mà bài này dùng, lấy thẳng từ nguồn.
- [LangGraph Persistence và Checkpoint](https://langchain-ai.github.io/langgraph/concepts/persistence/) — chi tiết về store Postgres/SQLite/Redis, namespace checkpoint và thread ID.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)` và pattern sửa trạng thái.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — pattern mà mọi agent LangGraph cài đặt; đọc để hiểu lý do của reasoning trace.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — nên chọn hình dạng đồ thị nào (chain, router, orchestrator-workers, evaluator-optimizer) và khi nào.
- Giai đoạn 11 · 09 (Function Calling) — primitive tool-call mà mọi nút agent LangGraph dùng lại.
- Giai đoạn 11 · 14 (Model Context Protocol) — khám phá tool bên ngoài cắm vào `ToolNode` của LangGraph qua adapter MCP.
- Giai đoạn 11 · 17 (Đánh đổi giữa các agent framework) — khi nào chọn LangGraph thay vì CrewAI, AutoGen hay Agno.

---

[← Bài trước: Prompt caching và cache ngữ cảnh](../../15-prompt-caching/docs/vi.md) · [Bài sau: Đánh đổi (trade-off) giữa các framework (khung) agent →](../../17-agent-framework-tradeoffs/docs/vi.md)
