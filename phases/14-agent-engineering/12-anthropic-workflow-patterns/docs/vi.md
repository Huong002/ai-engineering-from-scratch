# Các mẫu workflow của Anthropic: Đơn giản hơn là phức tạp

> Schluntz và Zhang (Anthropic, tháng 12/2024) phân biệt workflow (quy trình) — những đường đi định trước — với agent — hướng dùng công cụ động. Năm mẫu workflow bao phủ hầu hết các trường hợp. Hãy bắt đầu bằng gọi API trực tiếp. Chỉ thêm agent khi không thể đoán trước các bước.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent)
**Time:** ~60 phút

## Mục tiêu học tập

- Kể tên năm mẫu workflow của Anthropic: prompt chaining (chuỗi prompt), routing (định tuyến), parallelization, orchestrator-workers, evaluator-optimizer.
- Giải thích sự khác biệt giữa agent và workflow, cùng chi phí kỹ thuật của từng cách.
- Xác định khi nào nên chọn workflow thay vì agent (và ngược lại).
- Cài đặt cả năm mẫu bằng thư viện chuẩn, chạy trên một LLM được kịch bản hoá.

## Vấn đề

Các nhóm kỹ thuật vươn tới framework đa agent cho những bài toán chỉ cần một lời gọi hàm. Cái giá là thật: framework thêm các lớp che mất prompt, giấu luồng điều khiển và mời gọi sự phức tạp quá sớm. Bài viết tháng 12/2024 của Schluntz và Zhang là tiếng nói phản biện được trích dẫn nhiều nhất trong ngành: bắt đầu đơn giản, chỉ thêm phức tạp khi nó đáng giá.

## Khái niệm

### Workflow và agent

- **Workflow.** LLM và công cụ được orchestration (điều phối) qua các đường code định trước. Kỹ sư làm chủ đồ thị.
- **Agent.** LLM tự điều khiển công cụ và tự thực hiện các bước của mình một cách động. Mô hình làm chủ đồ thị.

Cả hai đều có chỗ đứng. Workflow rẻ hơn, nhanh hơn và dễ gỡ lỗi hơn. Agent mở ra các bài toán không có điểm dừng nhưng khiến các kiểu thất bại khó suy luận hơn.

### LLM được tăng cường

Nền tảng cho cả năm mẫu: một LLM với ba năng lực được nối vào — tìm kiếm (truy xuất), công cụ (hành động), bộ nhớ (lưu trữ). Bất kỳ lời gọi API nào cũng dùng được những thứ này.

### Năm mẫu

1. **Prompt chaining.** Đầu ra của lời gọi 1 là đầu vào của lời gọi 2. Dùng khi bài toán có thể tách tuyến tính gọn gàng. Có thể đặt các cổng kiểm tra bằng code giữa các bước.

2. **Routing.** Một LLM phân loại chọn LLM hoặc công cụ hạ nguồn để gọi. Dùng khi các đầu vào khác nhau về loại cần cách xử lý khác nhau (hỗ trợ cấp 1 so với hoàn tiền, báo lỗi, bán hàng).

3. **Parallelization.** Chạy N lời gọi LLM đồng thời, tổng hợp kết quả. Hai dạng: phân đoạn (các khối khác nhau) và bỏ phiếu (cùng prompt, N lần chạy, đa số/tổng hợp).

4. **Orchestrator-workers.** Một LLM điều phối quyết định động nên chạy worker nào (cũng là LLM) và tổng hợp đầu ra của chúng. Giống vòng lặp agent nhưng orchestrator không lặp vô hạn.

5. **Evaluator-optimizer.** Một LLM đề xuất câu trả lời, một LLM khác đánh giá nó. Lặp cho tới khi bộ đánh giá chấp nhận. Đây là Self-Refine (Bài 05) ở dạng tổng quát.

### Nơi workflow thắng agent

- **Bài toán đoán trước được.** Nếu bạn liệt kê được các bước, bạn nên liệt kê.
- **Bài toán bị ràng buộc chi phí.** Workflow có số bước hữu hạn; agent có thể xoáy vô tận.
- **Bài toán bị ràng buộc tuân thủ.** Kiểm toán viên muốn đọc đồ thị, chứ không muốn suy ra nó từ các vết chạy.

### Nơi agent thắng workflow

- **Nghiên cứu mở.** Khi bước tiếp theo phụ thuộc vào những gì bước trước trả về.
- **Bài toán độ dài thay đổi.** Công việc từ vài phút tới vài giờ với số bước chưa biết.
- **Lĩnh vực mới.** Khi bạn chưa biết workflow đúng là gì — khám phá trước, hệ thống hoá sau.

### Người bạn đồng hành: kỹ nghệ ngữ cảnh

"Effective context engineering for AI agents" (Anthropic 2025) chính thức hoá ngành học lân cận: cửa sổ 200k là một ngân sách, không phải một cái hộp. Cần đưa gì vào, khi nào nén gọn, khi nào để ngữ cảnh lớn lên. Được trình bày chi tiết trong bài học Giai đoạn 14 về nén ngữ cảnh (bài 06 trước đây của Giai đoạn 14 trong chương trình này, trước khi đánh lại số).

```figure
workflow-chain
```

## Tự tay làm

`code/main.py` cài đặt cả năm mẫu workflow trên một `ScriptedLLM`:

- `prompt_chain(input, steps)` — tuần tự.
- `route(input, classifier, handlers)` — phân loại + điều phối.
- `parallel_vote(prompt, n, aggregator)` — N lần chạy, tổng hợp.
- `orchestrator_workers(task, workers)` — orchestrator chọn worker.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — lặp cho tới khi đạt.

Chạy thử:

```
python3 code/main.py
```

Mỗi mẫu in ra vết chạy của nó. Tổng số dòng code mỗi mẫu khoảng 10-15; cái giá của một framework được đo bằng hàng nghìn dòng.

## Áp dụng

- Gọi API trực tiếp cho hầu hết công việc.
- Chỉ dùng framework khi mẫu thực sự cần trạng thái bền (LangGraph), đồng thời theo mô hình actor (AutoGen v0.4), hoặc tạo khuôn theo vai (CrewAI).
- Dùng Claude Agent SDK khi bạn muốn hình dạng harness của Claude Code mà không phải dựng lại nó.

## Đóng gói

`outputs/skill-workflow-picker.md` chọn đúng mẫu cho một mô tả công việc, kèm lý do quyết định và đường tái cấu trúc sang agent nếu workflow không đủ.

## Bài tập

1. Cài đặt routing với một ngưỡng tin cậy. Dưới ngưỡng -> chuyển cho người. Với một ca hỗ trợ cấp 1, ngưỡng nên đặt ở đâu?
2. Thêm timeout cho `parallel_vote`. Điều gì xảy ra khi một lời gọi bị treo? Bạn tổng hợp thế nào khi thiếu phiếu?
3. Biến `evaluator_optimizer` thành bandit: giữ top-2 đầu ra qua các vòng lặp để một kết quả tốt đến muộn không bị một kết quả tồi đến muộn ghi đè.
4. Kết hợp prompt chaining với routing: một router chọn một trong ba chuỗi. Đo chi phí token so với phương án một prompt lớn duy nhất.
5. Chọn một tính năng đang chạy thật của bạn. Vẽ đồ thị workflow. Đếm số bước. Liệu agent có thực sự tốt hơn ở đây?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Workflow | "Luồng định trước" | Đồ thị lời gọi LLM và công cụ do kỹ sư làm chủ |
| Agent | "AI tự chủ" | Đồ thị do mô hình làm chủ; điều khiển công cụ động |
| Augmented LLM | "LLM có công cụ" | LLM + tìm kiếm + công cụ + bộ nhớ; đơn vị nguyên tử |
| Prompt chaining | "Gọi tuần tự" | Đầu ra của lời gọi N là đầu vào của lời gọi N+1 |
| Routing | "Phân loại rồi điều phối" | Chọn chuỗi/mô hình nào xử lý đầu vào |
| Parallelization | "Toả ra song song" | N lời gọi đồng thời; tổng hợp bằng phân đoạn hoặc bỏ phiếu |
| Orchestrator-workers | "Agent điều phối" | LLM orchestrator chọn các LLM chuyên biệt một cách động |
| Evaluator-optimizer | "Bên đề xuất + bên chấm" | Lặp cho tới khi bộ đánh giá chấp nhận; Self-Refine tổng quát hoá |

## Đọc thêm

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — năm mẫu workflow
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — ngành học đồng hành
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — khi nào đồ thị có trạng thái đáng giá
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — mẫu orchestrator-workers, được sản phẩm hoá
