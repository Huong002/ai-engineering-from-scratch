# Benchmarks (bộ đánh giá): SWE-bench, GAIA, AgentBench

> Ba bộ đánh giá neo việc đánh giá agent trong năm 2026. SWE-bench kiểm tra việc vá mã. GAIA kiểm tra khả năng dùng công cụ của agent tổng quát. AgentBench kiểm tra suy luận đa môi trường. Hãy biết thành phần của chúng, câu chuyện contamination (nhiễm dữ liệu) của chúng, và những gì chúng không đo.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 06 (sử dụng công cụ)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu tên harness (khung kiểm thử) của SWE-bench (FAIL_TO_PASS) và giải thích vì sao nó lấy unit test làm cổng chặn.
- Giải thích vì sao SWE-bench Verified (OpenAI, 500 tác vụ) tồn tại và nó loại bỏ những gì.
- Mô tả thiết kế của GAIA: dễ với người, khó với AI; ba mức độ khó.
- Nêu tên tám môi trường của AgentBench và rào cản chính của nó với các LLM mã nguồn mở.
- Tóm tắt phát hiện nhiễm dữ liệu của SWE-bench+ và hàm ý của nó.

## Vấn đề

Bảng xếp hạng cho bạn biết mô hình nào thắng trên một bộ đánh giá. Chúng không cho bạn biết:

- Bộ đánh giá có bị nhiễm dữ liệu không (lời giải nằm trong dữ liệu huấn luyện, rò rỉ đề thi).
- Bộ đánh giá có đo đúng thứ bạn quan tâm không (mã so với duyệt web so với tổng quát).
- Bộ chấm có vững không (khớp AST, kiểm tra trạng thái, người duyệt).

Hãy biết ba bộ đánh giá neo này và các kiểu thất bại của chúng trước khi bạn trích dẫn một con số.

## Khái niệm

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- 2.294 issue GitHub thật từ 12 repo Python phổ biến.
- Agent nhận: cơ sở mã tại commit trước khi sửa + mô tả issue bằng ngôn ngữ tự nhiên.
- Agent tạo ra: một bản vá.
- Bộ chấm: áp bản vá, chạy bộ test của repo. Bản vá phải lật được các test FAIL_TO_PASS (trước đó fail, giờ pass) mà không làm hỏng các test PASS_TO_PASS.

SWE-agent (Yang et al., 2024) đạt 12.5% khi ra mắt nhờ nhấn mạnh giao diện agent-máy tính (lệnh trình soạn thảo file, cú pháp tìm kiếm mà mô hình hiểu được).

### SWE-bench Verified

OpenAI, tháng 8/2024. Tập con 500 tác vụ do người tuyển chọn. Loại bỏ các issue mơ hồ, test không đáng tin, và những tác vụ mà cách sửa không rõ. Bộ đánh giá chính cho câu hỏi "agent của bạn có giao được bản vá thật không?"

### Nhiễm dữ liệu

- Hơn 94% issue của SWE-bench có trước mốc cắt dữ liệu của hầu hết mô hình.
- **SWE-bench+** phát hiện 32.67% bản vá thành công làm lộ lời giải ngay trong nội dung issue (mô hình đã thấy cách sửa trong phần mô tả), và 31.08% đáng ngờ do độ phủ test yếu.
- Verified sạch hơn nhưng không phải không nhiễm.

Hàm ý thực tế: một mô hình đạt 50% trên SWE-bench có thể chỉ đạt 35% trên SWE-bench+. Luôn báo cả hai nếu bạn tuyên bố hiệu năng SWE-bench.

### GAIA (Mialon et al., Nov 2023)

- 466 câu hỏi; 300 câu được giữ cho bảng xếp hạng riêng tại huggingface.co/gaia-benchmark.
- Triết lý thiết kế: "đơn giản về khái niệm với người (92%) nhưng khó với AI (GPT-4 kèm plugin: 15%)."
- Kiểm tra suy luận, đa phương thức, web, dùng công cụ.
- Ba mức độ khó; Mức 3 đòi hỏi chuỗi công cụ dài qua nhiều phương thức.

GAIA là thứ bạn chạy để đo "năng lực tổng quát". Đừng nhầm với các bộ đánh giá riêng cho mã.

### AgentBench (Liu et al., ICLR 2024)

- 8 môi trường gồm mã (Bash, DB, KG), trò chơi (Alfworld, LTP), web (WebShop, Mind2Web), và sinh mở.
- Nhiều lượt, ~4k-13k lượt mỗi split.
- Phát hiện chính: suy luận dài hạn, ra quyết định, và tuân theo chỉ dẫn là những rào cản khiến LLM mã nguồn mở đuổi kịp mô hình thương mại.

### Những gì chúng không đo

- Chi phí vận hành thực tế (token, thời gian đồng hồ).
- Hành vi an toàn trong điều kiện đối kháng.
- Hiệu năng trên lĩnh vực của bạn (dùng eval riêng của bạn, Bài 30).
- Thất bại ở đuôi (bộ đánh giá lấy trung bình; người vận hành production quan tâm 1% tệ nhất).

### Nơi việc đánh giá đi sai

- **Cố chấp một con số.** SWE-bench 50% nói với bạn ít hơn phân bố chi phí + số bước P50/P75/P95.
- **Tuyên bố nhiễm.** Báo cáo SWE-bench mà không nhắc Verified hay SWE-bench+ là gây hiểu lầm.
- **Lấy bộ đánh giá làm mục tiêu phát triển.** Tối ưu cho bộ đánh giá đi lệch khỏi tính hữu ích trong production.

```figure
ae-swebench-gate
```

## Tự tay làm

`code/main.py` triển khai một harness kiểu SWE-bench đồ chơi:

- Tác vụ sửa lỗi tổng hợp (3 tác vụ).
- Một "agent" kịch bản hoá để đề xuất bản vá.
- Một trình chạy test kiểm tra FAIL_TO_PASS (lỗi giờ đã sửa) và PASS_TO_PASS (không có gì hỏng).
- Một bộ phân loại độ khó kiểu GAIA dựa trên độ sâu phân rã câu hỏi.

Chạy nó:

```
python3 code/main.py
```

Kết quả hiện tỉ lệ giải được theo từng tác vụ + theo độ khó và làm rõ các luật chấm.

## Áp dụng

- **SWE-bench Verified** cho agent mã. Luôn báo cáo điểm Verified.
- **GAIA** cho agent tổng quát. Dùng phần chia bảng xếp hạng riêng.
- **AgentBench** để so sánh đa môi trường.
- **Eval tự làm** (Bài 30) cho hình dạng thật của sản phẩm bạn.

## Đóng gói

`outputs/skill-benchmark-harness.md` dựng một harness kiểu SWE-bench cho bất kỳ cặp cơ sở mã-tác vụ nào với cổng FAIL_TO_PASS / PASS_TO_PASS.

## Bài tập

1. Chuyển harness đồ chơi sang chạy trên một repo thật (chọn một repo của bạn). Viết 3 test FAIL_TO_PASS cho các lỗi đã biết.
2. Thêm một chỉ số đếm số bước. Trên 3 tác vụ của bạn, mỗi lần giải cần bao nhiêu bước agent?
3. Đọc bài báo SWE-bench+. Triển khai một bước kiểm tra rò rỉ lời giải (so khớp mẫu nội dung issue với diff).
4. Tải một câu hỏi GAIA từ phần công khai. Lần theo những gì một agent cỡ GPT-4 sẽ làm. Nó cần những công cụ nào?
5. Đọc bảng phân tích theo từng môi trường của AgentBench. Môi trường nào giống bề mặt sản phẩm của bạn? "SOTA" ở đó trông như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| SWE-bench | "Bộ đánh giá agent mã" | 2.294 issue GitHub; bản vá phải lật được các test FAIL_TO_PASS |
| SWE-bench Verified | "SWE-bench sạch" | 500 tác vụ do người tuyển chọn, OpenAI |
| FAIL_TO_PASS | "Cổng sửa lỗi" | Các test trước đó fail, phải pass sau khi vá |
| PASS_TO_PASS | "Cổng không hồi quy" | Các test đã pass và phải vẫn pass |
| GAIA | "Bộ đánh giá tổng quát" | 466 câu hỏi đa công cụ dễ với người / khó với AI |
| AgentBench | "Bộ đánh giá đa môi trường" | 8 môi trường; nhiều lượt, tầm xa |
| Contamination | "Rò rỉ tập huấn luyện" | Tác vụ của bộ đánh giá có mặt trong dữ liệu huấn luyện mô hình |
| SWE-bench+ | "Kiểm toán nhiễm dữ liệu" | 32.67% rò rỉ lời giải được phát hiện trong các bản vá SWE-bench thành công |

## Đọc thêm

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — bộ đánh giá gốc
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — tập con tuyển chọn
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — bộ đánh giá tổng quát
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — bộ đánh giá đa môi trường
