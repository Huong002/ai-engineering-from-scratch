# Nhóm agent theo vai trò — Vai trò, tác vụ, quy trình

> Bốn thành phần cốt lõi: Agent, Task, Crew, Process. Hai cách tổ chức cấp cao: Crews (cộng tác tự chủ theo vai trò) và Flows (chạy theo sự kiện, có tính xác định). CrewAI là bản triển khai tham chiếu năm 2026, và tài liệu của họ nói thẳng: “với mọi ứng dụng sẵn sàng chạy trong môi trường thực tế, hãy bắt đầu bằng một Flow.”

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14 · Bài 12 (Các mẫu workflow), Giai đoạn 14 · Bài 14 (Mô hình Actor)
**Time:** ~75 phút

## Mục tiêu học tập

- Kể tên bốn thành phần cốt lõi của CrewAI (Agent, Task, Crew, Process) và trách nhiệm của từng thành phần.
- Phân biệt quy trình Sequential, Hierarchical và Consensus dự kiến; chọn một quy trình cho từng loại công việc.
- Phân biệt Crews (tự chủ theo vai trò) với Flows (chạy theo sự kiện, có tính xác định), và giải thích khuyến nghị của tài liệu cho môi trường thực tế.
- Kết nối công cụ bằng decorator `@tool` và lớp con của `BaseTool`; hiểu sự khác nhau giữa đầu ra có cấu trúc và văn bản tự do.
- Kể tên bốn loại memory của CrewAI và khi nào mỗi loại phát huy tác dụng.
- Triển khai một crew ba agent bằng stdlib (người nghiên cứu, người viết, biên tập viên) để tạo bản tóm lược.
- Nhận ra ba kiểu lỗi của CrewAI: prompt phình to, chi phí LLM quản lý, bàn giao dễ hỏng.

## Vấn đề

Các nhóm áp dụng framework đa agent đều gặp cùng một trở ngại. “Cộng tác tự chủ” nghe rất hấp dẫn trong bản demo. Nhưng rồi khách hàng báo lỗi và bạn cần chạy lại theo cách cho kết quả xác định. Hoặc bộ phận tài chính hỏi một crew do LLM điều phối tốn bao nhiêu tiền mỗi lần chạy. Hoặc người trực hệ thống cần biết agent nào bị kẹt lúc 3 giờ sáng.

Các crew được LLM điều phối tự do không trả lời rõ ràng được câu hỏi nào trong số đó. DAG thuần túy trả lời được tất cả, nhưng lại thiếu tính khám phá mà một agent động não cần có.

Cách CrewAI tách hai hình thức này nhìn thẳng vào sự đánh đổi. Dùng Crews cho công việc cộng tác, theo vai trò và mang tính khám phá. Dùng Flows cho hệ thống thực tế chạy theo sự kiện, do code kiểm soát và có thể kiểm tra lại. Cùng một framework, hai cách tổ chức; hãy chọn theo nhu cầu sử dụng.

## Khái niệm

### Bốn thành phần cốt lõi

Bề mặt API của CrewAI khá nhỏ. Hãy nhớ phần này; phần còn lại là cấu hình.

- **Agent.** `role + goal + backstory + tools + (optional) llm`. Backstory (bối cảnh nhân vật) có vai trò thiết yếu. Nó định hình giọng điệu, cách phán đoán và thời điểm agent dừng lại. Tools là các hàm agent có thể gọi (xem bên dưới).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. Một đơn vị công việc có thể tái sử dụng. `expected_output` là hợp đồng đầu ra. `context` liệt kê các tác vụ trước đó có đầu ra được chuyển vào. `output_pydantic` buộc đầu ra theo một cấu trúc nhất định.
- **Crew.** Vùng chứa. Nó quản lý danh sách `agents`, danh sách `tasks`, `process`, cùng các thiết lập `memory` + `verbose` + `manager_llm` tùy chọn.
- **Process.** Chiến lược thực thi. Sequential (tuần tự), Hierarchical (phân cấp), Consensus (đồng thuận, dự kiến). Nó quyết định cách tổ chức lượt chạy.

Các Agent không trực tiếp nhìn thấy nhau. Task tham chiếu đến Agent. Crew sắp xếp các Task theo trình tự. Process quyết định ai chọn tác vụ tiếp theo. Đó là toàn bộ mô hình tư duy bạn cần.

> **Đã kiểm chứng với** CrewAI 0.86 (2026-05). Các phiên bản mới hơn có thể đổi tên hoặc gộp các loại quy trình; hãy kiểm tra [tài liệu CrewAI Processes](https://docs.crewai.com/concepts/processes) trước khi dựa vào một cách tổ chức cụ thể.

### Sequential, Hierarchical và Consensus

- **Sequential.** Các tác vụ chạy theo thứ tự khai báo. Đầu ra của tác vụ N có thể được chuyển thành `context` cho tác vụ N+1. Chi phí thấp nhất. Dễ dự đoán nhất. Dùng khi thứ tự đã cố định.
- **Hierarchical.** Một Agent quản lý (một lần gọi LLM riêng) điều phối giữa các agent chuyên môn. CrewAI tạo agent quản lý từ cấu hình `manager_llm` của bạn hoặc từ cấu hình mặc định. Mỗi vòng, agent quản lý chọn tác vụ tiếp theo và có thể từ chối hoặc điều phối lại. Dùng khi bạn có từ bốn agent chuyên môn trở lên và thứ tự thực sự phụ thuộc vào đầu ra trước đó.
- **Consensus.** Mới nằm trong kế hoạch, chưa được triển khai trong API công khai. Tài liệu dành tên này cho một quy trình dựa trên biểu quyết trong tương lai. Hiện tại, đừng dựa vào nó.

Hierarchical thêm một lần gọi LLM mỗi vòng (cho agent quản lý), ngoài các lần gọi agent chuyên môn. Chi phí token có thể tăng gấp ba trong một lượt chạy năm bước. Chỉ chấp nhận chi phí đó khi bạn cần điều phối linh hoạt.

### Crews và Flows

Đây là cách tài liệu mở đầu chủ đề vào năm 2026.

- **Crew.** Tính tự chủ do LLM dẫn dắt. Framework quyết định cách chạy tại thời điểm thực thi. Phù hợp cho: nghiên cứu, động não, bản nháp đầu tiên, bất cứ nơi nào quá trình tìm đường là một phần của câu trả lời. Khó chạy lại để tái hiện kết quả. Khó kiểm thử. Tạo bản mẫu ít tốn công.
- **Flow.** Đồ thị chạy theo sự kiện do bạn kiểm soát. `@start` đánh dấu điểm bắt đầu. `@listen(topic)` đánh dấu một bước được kích hoạt khi bước khác phát ra topic đó. Mỗi bước là Python thông thường (có thể gọi Crew bên trong). Phù hợp cho: hệ thống thực tế. Quan sát được. Kiểm thử được. Có tính xác định.

Khuyến nghị của tài liệu cho hệ thống thực tế năm 2026: bắt đầu bằng một Flow. Đưa Crews vào dưới dạng lời gọi `Crew.kickoff()` bên trong các bước của Flow khi tính tự chủ xứng đáng với chi phí bỏ ra. Flow cho bạn dấu vết để kiểm tra, Crew cho bạn khả năng khám phá. Hãy kết hợp, đừng chỉ chọn một.

### Tích hợp công cụ

Có ba cách cấp công cụ cho Agent. Hãy chọn cách đơn giản nhất đáp ứng nhu cầu.

1. **Decorator `@tool`.** Biến hàm thuần thành công cụ. Chữ ký hàm là schema; docstring là phần mô tả LLM nhìn thấy. Phù hợp nhất cho các hàm hỗ trợ dùng một lần.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **Lớp con của `BaseTool`.** Công cụ dạng lớp với schema đối số tường minh, hỗ trợ bất đồng bộ và thử lại. Dùng khi công cụ có trạng thái (client, cache) hoặc cần đối số có cấu trúc.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Bộ công cụ tích hợp sẵn.** CrewAI cung cấp các adapter chính thức: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. Chỉ cần một lệnh import để kết nối.

Đầu ra có cấu trúc dùng Pydantic. Truyền `output_pydantic=MyModel` cho Task. CrewAI kiểm tra phản hồi của LLM theo model và sẽ chuyển đổi hoặc thử lại. Kết hợp cách này với chuỗi `expected_output` rõ ràng. Đầu ra văn bản tự do phù hợp cho bản nháp; đầu ra có cấu trúc mới là thứ các Flow phía sau có thể sử dụng.

### Các cơ chế memory

CrewAI tích hợp sẵn bốn loại memory. Chúng có thể kết hợp với nhau: một Crew có thể bật cả bốn cùng lúc.

> **Đã kiểm chứng với** CrewAI 0.86 (2026-05). Các bản phát hành gần đây đưa mọi thứ qua một hệ thống `Memory` thống nhất, bao bọc bốn nơi lưu trữ này. Mô hình khái niệm bên dưới vẫn đúng, nhưng API công khai có thể được gom vào một điểm truy cập `Memory` duy nhất ở các phiên bản mới hơn; hãy xem [tài liệu memory của CrewAI](https://docs.crewai.com/concepts/memory) để biết API hiện tại.

- **Short-term (ngắn hạn).** Bộ đệm hội thoại trong một lượt chạy. Bị xóa khi lượt chạy kết thúc.
- **Long-term (dài hạn).** Được lưu qua nhiều lượt chạy. Lưu trong cơ sở dữ liệu vector (mặc định là Chroma, có thể thay thế). Được truy xuất theo mức độ tương đồng với tác vụ hiện tại.
- **Entity (theo thực thể).** Thông tin về từng thực thể. “Khách hàng X đang dùng gói enterprise.” Tra cứu theo thực thể, không theo độ tương đồng. Tồn tại qua nhiều lượt chạy.
- **Contextual (theo ngữ cảnh).** Truy xuất khi tập hợp thông tin cho tác vụ. Lấy memory liên quan đúng lúc Agent cần, thay vì tải sẵn.

Bật trên Crew bằng `memory=True` hoặc cấu hình theo từng loại. Memory dùng nhà cung cấp embedding do bạn cấu hình (mặc định là OpenAI, có thể đổi sang giải pháp chạy cục bộ). Memory là một điểm khiến CrewAI đáng dùng hơn các framework gọn nhẹ hơn; với LangGraph thuần túy, bạn phải tự kết nối từng loại này.

### Khi nào nhóm theo vai trò phù hợp

- Ba đến sáu agent có vai trò rõ ràng và quy trình cộng tác. Soạn thảo, rà soát, lập kế hoạch, động não.
- Công việc điều phối mà phán đoán của LLM về bước tiếp theo là một phần giá trị mang lại (Hierarchical).
- Bất cứ nơi nào nhóm thấy `role + goal + backstory` dễ đọc hơn định nghĩa đồ thị.

### Khi nào không phù hợp

- DAG có tính xác định và thứ tự nghiêm ngặt. Dùng LangGraph (Bài 13). Cấu trúc đồ thị là cách biểu diễn phù hợp; cách tổ chức theo vai trò của CrewAI chỉ gây thêm vướng víu.
- Giới hạn độ trễ dưới một giây. Hierarchical thêm các lượt trao đổi. Ngay cả Sequential cũng chạy nối tiếp các prompt chứa backstory và đầu ra trước đó.
- Vòng lặp một agent. Bỏ qua framework; một vòng lặp agent (Bài 1) cùng bộ công cụ đã đủ gọn.

Bài 17 (Đánh đổi giữa các framework agent) trình bày điều này trong một ma trận. Nói ngắn gọn: CrewAI nằm ở nhóm “cộng tác theo vai trò”.

### Phụ thuộc và môi trường

Không phụ thuộc vào LangChain. Python 3.10 đến 3.13. Dùng `uv`. Số sao: xem [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) (số liệu tại thời điểm 2026-05). Tích hợp AWS Bedrock có tài liệu hướng dẫn; các phép đo do nhà cung cấp báo cáo cho thấy tốc độ tăng đáng kể so với LangGraph trên tác vụ QA, nhưng phương pháp đo (tập dữ liệu, phần cứng, chỉ số đánh giá) không được công bố, nên chỉ xem các con số từ nhà cung cấp framework là tham khảo về xu hướng.

### Khi nào cách làm này gặp lỗi

- **Prompt phình to vì backstory.** Một backstory dài 2000 từ cho mỗi agent trong crew năm agent sẽ tiêu tốn ngân sách context trước cả lần gọi công cụ đầu tiên. Giữ backstory dưới 200 từ. Tái sử dụng cách diễn đạt giữa các agent; đừng lặp lại quy chuẩn văn phong năm lần.
- **Chi phí token cho LLM quản lý.** Quy trình Hierarchical thêm một lần gọi LLM quản lý trước mỗi lần gọi agent chuyên môn. Với crew năm tác vụ, đó là sáu lần gọi LLM thay vì năm, và lần gọi agent quản lý còn mang theo toàn bộ danh sách tác vụ cùng các đầu ra trước đó. Chuyển sang Sequential trừ khi việc điều phối phụ thuộc vào đầu ra.
- **Bàn giao dễ hỏng.** `expected_output` của Task N là “một dàn ý”. Task N+1 đọc nó qua `context` và cố phân tích thành ba phần. LLM lại tạo ra bốn phần. Agent phía sau phải tự ứng biến. Khắc phục bằng `output_pydantic` trên Task N để Task N+1 đọc một đối tượng có kiểu, không phải văn bản tự do.
- **Đưa Crew thẳng vào hệ thống thực tế.** Một Crew tự do được triển khai mà không có Flow bao ngoài. Đầu ra biến động nhiều; không thể chạy lại để tái hiện; người trực hệ thống không thể so sánh một lượt chạy lỗi với một lượt chạy tốt. Hãy bao nó trong một Flow.

```figure
ae-crew-vs-flow
```

## Tự tay làm

`code/main.py` triển khai cả hai cách tổ chức bằng stdlib, cùng một crew ba agent.

Cấu trúc:

- Các dataclass `Agent`, `Task` tương ứng với API của CrewAI.
- `SequentialCrew.kickoff(inputs)` chạy các tác vụ theo thứ tự khai báo, chuyển đầu ra tiếp nối nhau dưới dạng `context`.
- `HierarchicalCrew.kickoff(topic)` thêm một Agent quản lý chọn agent chuyên môn tiếp theo ở mỗi vòng, dừng khi nhận “done”.
- `Flow` với các decorator `@start` và `@listen(topic)`, một vòng lặp sự kiện nhỏ và dấu vết thực thi.
- Decorator `tool(name)` mô phỏng cách dùng `@tool` của CrewAI.
- `Memory` với các nơi lưu trữ `short_term`, `long_term`, `entity`; phép đo độ tương đồng giả lập dùng numpy.
- Phản hồi LLM giả lập là các chuỗi viết sẵn, được chọn theo vai trò và tiền tố đầu vào. Không dùng mạng. Kết quả có tính xác định.

Demo cụ thể: crew gồm người nghiên cứu, người viết và biên tập viên tạo bản tóm lược về “agent engineering 2026”. Người nghiên cứu lấy các nguồn (giả lập). Người viết soạn bản nháp. Biên tập viên chỉnh gọn. Cùng crew đó chạy qua một Flow để cho thấy cách tổ chức có tính xác định.

Chạy như sau:

```bash
python3 code/main.py
```

Dấu vết thực thi bao gồm: crew Sequential chuyển đầu ra qua `context`, crew Hierarchical với các lựa chọn của agent quản lý (người nghiên cứu, người viết, biên tập viên, rồi “done”), Flow chạy cùng ba bước với các topic tường minh (`researched`, `drafted`, `edited`), các lần gọi công cụ được điều hướng qua `@tool`, và memory dài hạn còn tồn tại qua hai lần kickoff.

Dấu vết của Crew linh hoạt; về nguyên tắc, agent quản lý có thể đổi thứ tự. Dấu vết của Flow thì cố định. Lựa chọn đó chính là trọng tâm bài học.

## Áp dụng

- **CrewAI Flow** cho hệ thống thực tế. Ngay cả khi Flow chỉ có một bước gọi `Crew.kickoff()`. Flow tạo ranh giới để kiểm tra.
- **CrewAI Crew (Sequential)** cho công việc cộng tác có thứ tự rõ ràng, đặc biệt là soạn bản nháp đầu tiên và các vòng rà soát.
- **CrewAI Crew (Hierarchical)** khi việc điều phối phụ thuộc vào đầu ra và bạn có từ bốn agent chuyên môn trở lên.
- **LangGraph** (Bài 13) cho máy trạng thái tường minh, khả năng tiếp tục sau gián đoạn và thứ tự nghiêm ngặt.
- **AutoGen v0.4** (Bài 14) cho xử lý đồng thời theo mô hình actor và cô lập lỗi.
- **OpenAI Agents SDK** (Bài 16) cho sản phẩm ưu tiên OpenAI, có bàn giao và rào chắn an toàn.
- **Claude Agent SDK** (Bài 17) cho sản phẩm ưu tiên Claude, có subagent và nơi lưu session.

## Đóng gói

`outputs/skill-crew-or-flow.md` chọn Crew hay Flow cho một tác vụ và dựng bộ khung triển khai tối thiểu. Từ chối dứt khoát Crew không có backstory, Flow không có topic tường minh, và Hierarchical có dưới ba agent chuyên môn.

## Những điểm dễ mắc lỗi

- **Xem backstory chỉ là phần trang trí.** Nó định hình đầu ra. Thử ba biến thể cho mỗi agent; mức độ khác biệt là có thật. Chọn một bản rồi cố định nó.
- **Bỏ qua `expected_output`.** Không có hợp đồng cho từng tác vụ, các tác vụ phía sau sẽ nhận bất cứ thứ gì LLM tạo ra. Crew vẫn chạy; việc kiểm tra lại thì thất bại.
- **Luôn bật memory.** Memory dài hạn ghi dữ liệu sau mỗi lượt chạy. Cơ sở dữ liệu vector lớn dần. Kết quả truy xuất trở nên nhiễu. Chỉ ghi từ những tác vụ tạo ra thông tin cần lưu lâu dài.
- **Prompt của agent quản lý bị lệch.** Prompt của agent quản lý trong Hierarchical được tạo ngầm. Nếu việc điều phối trở nên bất thường, hãy xuất prompt ở chế độ verbose và đọc nó.
- **Công cụ có tác dụng phụ trong Crews.** Crew có thể gọi một công cụ nhiều lần hơn dự kiến. POST, DELETE và thanh toán phải nằm trong một bước Flow, không bao giờ nằm trong công cụ của Crew.

## Bài tập

1. Chuyển crew Sequential thành Flow. Đếm những điểm mà độ biến động giảm xuống. Ghi lại chỗ nào trở nên khó đọc hơn.
2. Thêm entity memory vào crew: thông tin về một khách hàng được giữ qua nhiều lần kickoff. Kiểm tra xem truy xuất có lấy đúng thực thể không.
3. Triển khai quy trình Hierarchical trong đó agent quản lý từ chối điều phối cho biên tập viên cho đến khi đầu ra của người viết có ít nhất ba đoạn. Theo dõi lần thử lại.
4. Kết nối một lớp con `BaseTool` để tìm kiếm web (giả lập). So sánh cấu trúc dấu vết thực thi với phiên bản dùng decorator `@tool`.
5. Thêm `output_pydantic=Brief` vào tác vụ của biên tập viên, trong đó `Brief` có `title`, `summary`, `sections`. Cho tác vụ của người viết tạo JSON sai định dạng một lần; kiểm tra hành vi thử lại của CrewAI trong dấu vết thực thi.
6. Đọc phần giới thiệu tài liệu CrewAI. Chuyển ví dụ nhỏ này sang API `crewai` thật. Phiên bản stdlib đã bỏ qua những đảm bảo nào?
7. Kết nối AgentOps hoặc Langfuse (Bài 24) với một lượt chạy thật. Phiên bản stdlib đã bỏ sót những dấu vết thực thi nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Agent | “Nhân vật” | Vai trò + mục tiêu + backstory + công cụ |
| Task | “Đơn vị công việc” | Mô tả + đầu ra mong đợi + agent được giao + đầu ra có cấu trúc tùy chọn |
| Crew | “Nhóm agent” | Vùng chứa Agent + Task + Process |
| Process | “Chiến lược thực thi” | Sequential / Hierarchical / Consensus (dự kiến) |
| Flow | “Luồng tất định” | Theo sự kiện, do code quản lý, kiểm thử được |
| Backstory | “Prompt mô tả nhân vật” | Yếu tố định hình giọng điệu và cách phán đoán của Agent |
| `@tool` | “Công cụ dạng hàm” | Decorator biến một hàm thành công cụ Agent có thể gọi |
| `BaseTool` | “Công cụ dạng lớp” | Công cụ dạng lớp có schema đối số, khả năng thử lại và hỗ trợ bất đồng bộ |
| Entity memory | “Thông tin theo từng thực thể” | Memory giới hạn theo khách hàng / tài khoản / vấn đề |
| Long-term memory | “Memory qua nhiều lượt chạy” | Memory dựa trên vector, còn tồn tại giữa các lần kickoff |
| Contextual memory | “Truy xuất đúng lúc” | Memory được lấy vào thời điểm Agent cần |
| Manager LLM | “Agent điều phối” | LLM bổ sung trong quy trình Hierarchical, chọn tác vụ tiếp theo |
| `expected_output` | “Hợp đồng tác vụ” | Chuỗi cho Agent (và bộ phận kiểm tra) biết phải trả về đầu ra theo cấu trúc nào |

## Đọc thêm

- [Phần giới thiệu tài liệu CrewAI](https://docs.crewai.com/en/introduction): các khái niệm và hướng triển khai được khuyến nghị cho hệ thống thực tế
- [Hướng dẫn CrewAI Flows](https://docs.crewai.com/en/concepts/flows): cách tổ chức theo sự kiện, `@start`, `@listen`
- [Tài liệu tham khảo về công cụ CrewAI](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, các bộ công cụ tích hợp sẵn
- [Memory của CrewAI](https://docs.crewai.com/en/concepts/memory): ngắn hạn, dài hạn, theo thực thể, theo ngữ cảnh
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): khi nào đa agent có ích và khi nào không
- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview): phương án thay thế dựa trên máy trạng thái

---

[← Bài trước: Mô hình actor (actor model) cho agent (tác tử)](../../14-autogen-actor-model/docs/vi.md) · [Bài sau: OpenAI Agents SDK: Handoff, Guardrail, Tracing →](../../16-openai-agents-sdk/docs/vi.md)
