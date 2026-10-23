# ReWOO và Plan-and-Execute: Tách rời lập kế hoạch

> ReAct xen kẽ suy nghĩ và hành động trong một luồng. ReWOO tách chúng ra: một kế hoạch lớn đặt trước, rồi thực thi. Ít hơn 5 lần token, +4% độ chính xác trên HotpotQA, và bạn có thể distill planner thành một model 7B. Plan-and-Execute (lập kế hoạch rồi thực thi) đã khái quát hoá nó; Plan-and-Act mở rộng nó sang điều hướng web.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích vì sao việc tách Planner / Worker / Solver của ReWOO tiết kiệm token và tăng độ bền so với vòng lặp xen kẽ của ReAct.
- Cài đặt một DAG kế hoạch, một executor chạy theo thứ tự phụ thuộc, và một solver ghép đầu ra của worker — tất cả bằng stdlib.
- Quyết định khi nào một tác vụ nên chạy theo kiểu plan-then-execute thay vì ReAct xen kẽ, dùng cách đóng khung "năm mẫu workflow" năm 2026 (Anthropic).
- Nhận ra khi nào cần dữ liệu kế hoạch tổng hợp của Plan-and-Act cho các tác vụ web hoặc di động dài hạn.

## Vấn đề

Vòng lặp xen kẽ suy nghĩ-hành động-quan sát của ReAct đơn giản và linh hoạt, nhưng mỗi lần gọi công cụ phải mang theo toàn bộ ngữ cảnh trước đó — gồm cả mọi suy nghĩ trước. Lượng token tăng theo bình phương độ sâu. Tệ hơn: khi một công cụ lỗi giữa vòng lặp, model phải suy ra lại toàn bộ kế hoạch từ quan sát lỗi.

ReWOO (Xu et al., arXiv:2305.18323, tháng 5 năm 2023) nhận ra điều này và đặt cược: lập kế hoạch toàn bộ từ đầu, lấy evidence (bằng chứng) song song, ghép câu trả lời ở cuối. Một lần gọi LLM để lập kế hoạch, N lần gọi công cụ để lấy evidence, một lần gọi LLM để giải. Cái giá là ít linh hoạt hơn (kế hoạch tĩnh) để đổi lấy hiệu quả token tốt hơn nhiều và chế độ lỗi rõ ràng hơn.

## Khái niệm

### Ba vai trò

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner tạo ra một DAG. Mỗi nút nêu tên một công cụ, tham số của nó, và nó phụ thuộc vào những nút nào trước đó (tham chiếu như `#E1`, `#E2`). Worker thực thi các nút theo thứ tự topo. Solver khâu mọi thứ lại với nhau.

### Vì sao ít token hơn 5 lần

ReAct làm độ dài prompt tăng tuyến tính theo số bước. Ở bước 10, prompt chứa suy nghĩ 1 cộng hành động 1 cộng quan sát 1 cộng suy nghĩ 2 cộng hành động 2 cộng quan sát 2, và cứ thế. Mỗi bước trung gian cũng lặp lại một cách dư thừa prompt ban đầu.

ReWOO trả giá bằng một prompt cho planner (lớn), N prompt nhỏ cho worker (mỗi cái chỉ là lời gọi công cụ, không có chuỗi), và một prompt cho solver. Trên HotpotQA, bài báo đo được ít hơn khoảng 5 lần token mà vẫn đạt +4 độ chính xác tuyệt đối.

### Vì sao nó bền hơn

Nếu worker 3 lỗi trong ReAct, vòng lặp phải suy luận thoát khỏi lỗi ngay giữa dòng chảy. Trong ReWOO, worker 3 trả về một chuỗi lỗi; solver thấy nó trong ngữ cảnh cùng kế hoạch ban đầu và có thể suy giảm một cách êm ái. Định vị lỗi là theo từng nút, không phải theo từng bước.

### Distill planner

Kết quả thứ hai của bài báo: vì planner không thấy các quan sát, bạn có thể fine-tune một model 7B trên đầu ra planner của một teacher 175B. Model nhỏ lo việc lập kế hoạch; không cần model lớn khi suy luận. Điều này giờ đã thành chuẩn — nhiều agent production năm 2026 dùng planner nhỏ và executor lớn hoặc ngược lại.

### Plan-and-Execute (2023)

Bài viết tháng 8 năm 2023 của nhóm LangChain khái quát hoá ReWOO thành một tên mẫu: Plan-and-Execute. Planner đặt trước phát ra một danh sách bước, executor chạy từng bước, một replanner (lập lại kế hoạch) tuỳ chọn có thể sửa lại sau khi quan sát kết quả. Cách này gần với ReAct hơn ReWOO (replanner đưa quan sát trở lại việc lập kế hoạch) nhưng vẫn giữ được phần tiết kiệm token.

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act mở rộng mẫu này cho các agent web và di động dài hạn. Đóng góp chính là dữ liệu kế hoạch tổng hợp: một bộ sinh quỹ đạo có nhãn tạo ra dữ liệu huấn luyện trong đó kế hoạch là tường minh. Nó được dùng để fine-tune các model planner vẫn làm việc được qua 30–50 bước trên các tác vụ kiểu WebArena, nơi một quỹ đạo ReAct đơn lẻ mất mạch.

### Khi nào chọn cái nào

| Mẫu | Khi nào |
|---------|------|
| ReAct | Tác vụ ngắn, môi trường chưa biết, cần xử lý ngoại lệ phản ứng nhanh |
| ReWOO | Tác vụ có cấu trúc với công cụ đã biết, nhạy cảm token, bằng chứng song song hoá được |
| Plan-and-Execute | Giống ReWOO nhưng có lập lại kế hoạch sau khi thực thi một phần |
| Plan-and-Act | Dài hạn (>30 bước), web/di động/điều khiển máy tính |
| Tree of Thoughts | Tìm kiếm đáng để trả giá (Bài 04) |

Hướng dẫn tháng 12 năm 2024 của Anthropic: bắt đầu với cái đơn giản nhất. Nếu tác vụ chỉ là một lần gọi công cụ cộng một bản tóm tắt, đừng xây ReWOO. Nếu tác vụ là một bài nghiên cứu 40 bước, đừng chỉ làm ReAct.

```figure
rewoo-plan
```

## Tự tay làm

`code/main.py` cài đặt một ReWOO đồ chơi:

- `Planner` — một policy kịch bản phát ra DAG kế hoạch từ một prompt.
- `Worker` — gửi lời gọi công cụ của từng nút qua registry.
- `Solver` — ghép theo kịch bản, đọc evidence và tạo ra câu trả lời cuối.
- Giải phụ thuộc — các tham chiếu như `#E1` được thay bằng đầu ra của worker trước đó.

Bản demo trả lời câu "Dân số của thủ đô nước Pháp là bao nhiêu, làm tròn đến triệu?" bằng một kế hoạch hai bước: (1) tra thủ đô, (2) tra dân số, rồi giải.

Chạy nó:

```
python3 code/main.py
```

Vết chạy hiển thị toàn bộ kế hoạch trước, rồi kết quả worker, rồi phần ghép của solver. So sánh số token (chúng ta in một con số ký tự ước lượng) với một lượt chạy xen kẽ kiểu ReAct — ReWOO thắng ở loại tác vụ có cấu trúc này.

## Áp dụng

LangGraph có sẵn Plan-and-Execute như một công thức (`create_react_agent` cho ReAct, graph tuỳ chỉnh cho plan-execute). Flows của CrewAI mã hoá mẫu này trực tiếp: bạn định nghĩa các tác vụ từ đầu và DAG của Flow thực thi chúng. Cách tiếp cận dữ liệu tổng hợp của Plan-and-Act vẫn chủ yếu là nghiên cứu; mẫu lúc chạy (DAG kế hoạch tường minh) đã có trong production qua LangGraph và CrewAI Flows.

## Đóng gói

`outputs/skill-rewoo-planner.md` tạo một DAG kế hoạch ReWOO từ yêu cầu người dùng, với một danh mục công cụ cho trước. Nó kiểm tra kế hoạch (không chu trình, mọi tham chiếu được giải, mọi công cụ đều tồn tại) trước khi chuyển cho executor.

## Bài tập

1. Song song hoá việc thực thi worker cho các nút kế hoạch độc lập. Nó mang lại gì trên một DAG 6 nút với 2 nhóm song song?
2. Thêm một nút replanner kích hoạt nếu bất kỳ worker nào trả về lỗi. Thay đổi nhỏ nhất nào trên ReWOO biến nó thành Plan-and-Execute?
3. Thay `Planner` bằng một model nhỏ (cỡ 7B) và giữ `Solver` trên một model frontier. So sánh chất lượng đầu-cuối — cách tách này hỏng ở đâu?
4. Đọc Mục 4 của bài báo ReWOO về distill planner. Tái hiện kết quả 175B -> 7B ở mức khái niệm: bạn cần dữ liệu huấn luyện gì, và chấm điểm chất lượng kế hoạch thế nào?
5. Chuyển bản đồ chơi sang dạng quỹ đạo của Plan-and-Act: kế hoạch là một chuỗi, không phải DAG. Những đánh đổi nào thay đổi?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| ReWOO | "Suy luận không có quan sát" | Lập kế hoạch, rồi lấy bằng chứng song song, rồi giải — không có quan sát trong prompt lập kế hoạch |
| Plan-and-Execute | "Mẫu plan-execute của LangChain" | ReWOO có thêm một nút replanner tuỳ chọn sau khi thực thi |
| Plan-and-Act | "Plan-execute mở rộng" | Tách planner/executor tường minh với dữ liệu huấn luyện kế hoạch tổng hợp cho tác vụ dài hạn |
| Evidence reference | "#E1, #E2, ..." | Chỗ giữ chỗ trong nút kế hoạch được thay bằng đầu ra worker trước đó lúc gửi đi |
| Planner distillation | "Planner nhỏ, executor lớn" | Fine-tune một model nhỏ trên vết planner từ một teacher lớn |
| Token efficiency | "Ít lượt đi-về hơn" | Ít hơn 5 lần token trên HotpotQA so với ReAct trong bài báo |
| DAG executor | "Bộ gửi theo topo" | Chạy các nút kế hoạch theo thứ tự phụ thuộc; song song ở mỗi mức |

## Đọc thêm

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — bài báo gốc
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — planner-executor mở rộng với kế hoạch tổng hợp
- [Hướng dẫn Plan-and-Execute của LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — công thức của framework
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — chọn mẫu đơn giản nhất mà vẫn hiệu quả
