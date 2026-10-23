# Tree of Thoughts (cây tư duy) và LATS: Tìm kiếm có chủ đích

> Một quỹ đạo chain-of-thought duy nhất không có chỗ cho backtracking (quay lui). ToT (Yao et al., 2023) biến việc suy luận thành một cây với tự đánh giá trên từng nút. LATS (Language Agent Tree Search — tìm kiếm cây cho tác nhân ngôn ngữ; Zhou et al., 2024) hợp nhất ToT với ReAct và Reflexion dưới Monte Carlo Tree Search. Game of 24 đi từ 4% (CoT) lên 74% (ToT); LATS đạt 92.7% pass@1 trên HumanEval.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 03 (Reflexion)
**Time:** ~75 phút

## Mục tiêu học tập

- Đặt suy luận dưới góc nhìn tìm kiếm: nút là "thought", cạnh là "mở rộng", giá trị là "mức hứa hẹn".
- Cài đặt tìm kiếm cây kiểu ToT BFS chỉ bằng stdlib, có chấm điểm tự đánh giá.
- Mở rộng thành một vòng MCTS kiểu LATS đồ chơi với select / expand / simulate / backpropagate.
- Quyết định khi nào tìm kiếm đáng với chi phí token nhân lên (Game of 24, sinh mã) và khi nào một quỹ đạo là đủ (hỏi đáp đơn giản).

## Vấn đề

Chain-of-thought là một đường đi tuyến tính. Nếu bước đầu sai, mọi bước sau đều dựa trên một tiền đề tồi. Trên Game of 24 (dùng bốn chữ số với + − × ÷ để ra 24), GPT-4 CoT chỉ đạt 4% độ chính xác. Mô hình chọn sai biểu thức con từ sớm và không thể gỡ lại.

Điều mà suy luận cần là khả năng đề xuất nhiều ứng viên, đánh giá chúng, chọn những cái hứa hẹn, và quay lui khi gặp ngõ cụt. Đó là tìm kiếm. Tree of Thoughts và LATS là hai công thức kinh điển.

## Khái niệm

### Tree of Thoughts (Yao et al., NeurIPS 2023)

Mỗi nút là một bước trung gian mạch lạc ("một thought"). Mỗi nút có thể mở rộng thành K thought con. LLM tự đánh giá từng nút bằng một prompt chấm điểm. Tìm kiếm khám phá cây — BFS, DFS hoặc beam.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Tự đánh giá là phần chịu lực. Bài báo đưa ra ba biến thể: phân loại `sure / likely / impossible`, điểm số `1..10`, và bỏ phiếu giữa các ứng viên. Cả ba đều vượt CoT đáng kể trên Game of 24 (4% -> 74% với GPT-4).

### LATS (Zhou et al., ICML 2024)

LATS hợp nhất ToT, ReAct và Reflexion dưới MCTS. LLM đóng ba vai:

- **Policy**: đề xuất các hành động kế tiếp ứng viên (kiểu ReAct).
- **Value function** (hàm giá trị): chấm điểm một phần quỹ đạo (tự đánh giá kiểu ToT).
- **Self-reflector**: khi thất bại, viết một reflection bằng ngôn ngữ tự nhiên (kiểu Reflexion) và dùng nó để gieo lại các rollout (mô phỏng quỹ đạo) sau này.

Phản hồi từ môi trường (observation) được trộn vào hàm giá trị để tìm kiếm dựa trên kết quả công cụ thật, không chỉ ý kiến của mô hình. Kết quả tại thời điểm bài báo: HumanEval pass@1 92.7% với GPT-4 (SOTA), WebShop trung bình 75.9 với GPT-3.5 (tiến gần fine-tuning dựa trên gradient).

### MCTS, ở mức tối thiểu

Bốn pha mỗi vòng lặp:

1. **Select** — đi từ gốc tới một lá bằng UCT (upper confidence bound for trees).
2. **Expand** — sinh K nút con qua policy.
3. **Simulate** — rollout từ một nút con bằng policy, chấm điểm lá bằng hàm giá trị (hoặc phần thưởng từ môi trường).
4. **Backpropagate** — cập nhật số lần thăm và ước lượng giá trị ngược lên theo đường đi.

Công thức UCT: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Số hạng đầu là khai thác; số hạng hai là khám phá. Tinh chỉnh `c` theo từng bài toán.

### Thực tế chi phí

Tìm kiếm làm nổ token. ToT trên Game of 24 dùng gấp 100–1000 lần token của CoT. LATS cũng tương tự. Đây không phải miễn phí; hãy dành tìm kiếm cho:

- Bài toán mà một quỹ đạo duy nhất đã chứng minh là không đủ (Game of 24, mã phức tạp).
- Bài toán mà thời gian thực ít quan trọng hơn tính đúng.
- Bài toán có hàm giá trị rẻ và đáng tin (unit test cho mã, mục tiêu rõ ràng cho toán).

Nếu bài toán của bạn có một đáp án đúng duy nhất và bộ đánh giá nhiễu, tìm kiếm thường làm mọi thứ tệ hơn — nó tìm ra một đáp án sai nhưng "điểm cao".

### Vị thế năm 2026

Hầu hết agent trong production không chạy LATS. Chúng chạy ReAct với kiểm chứng bám vào công cụ (CRITIC, Bài 05). Tìm kiếm xuất hiện trong các ngách chuyên biệt:

- Agent lập trình dùng test làm hàm giá trị (kiểu HumanEval).
- Agent nghiên cứu sâu khám phá nhiều đường truy vấn.
- Quy trình nặng về lập kế hoạch bên trong subgraph của LangGraph.

AlphaEvolve (Bài 11) là cực đoan của 2025: tìm kiếm tiến hoá trên mã, độ thích nghi kiểm chứng được bằng máy, thu hoạch ở biên (cải thiện nhân ma trận 4x4 đầu tiên sau 56 năm).

```figure
tree-of-thoughts
```

## Tự tay làm

`code/main.py` triển khai:

- Một ToT BFS nhỏ trên bài toán "chọn phép toán" được cách điệu.
- Một vòng MCTS kiểu LATS đồ chơi trên cùng bài toán (Select / Expand / Simulate / Backpropagate) với chọn lựa bằng UCT.
- Một hàm giá trị kết hợp điểm tượng trưng với điểm tự đánh giá.

Chạy nó:

```
python3 code/main.py
```

Trace cho thấy ToT mở rộng ba ứng viên mỗi nút bằng BFS, so với LATS hội tụ về rollout tốt nhất qua MCTS. Số token được in cho cả hai.

## Áp dụng

LangGraph cung cấp khám phá kiểu ToT dưới dạng các mẫu subgraph; bài blog của nhóm LangChain về LATS (tháng 5/2024) là hướng dẫn tham chiếu. LlamaIndex có một agent `TreeOfThoughts`. Với hầu hết agent production năm 2026, mẫu này nằm sau một cổng `if task_complexity > threshold: use_search()` — xem mẫu evaluator-optimizer trong Bài 05.

## Đóng gói

`outputs/skill-search-policy.md` chọn giữa ReAct tuyến tính, ToT, LATS và tìm kiếm tiến hoá dựa trên hình dạng bài toán, ngân sách và độ tin cậy của bộ đánh giá.

## Bài tập

1. Chạy LATS đồ chơi với UCT c=0.1 so với c=2.0. Trace thay đổi thế nào?
2. Đổi hàm giá trị sang một bộ chấm điểm nhiễu hơn (thêm nhiễu ngẫu nhiên). MCTS còn tìm được lá tốt nhất không? Tỷ lệ tín hiệu trên nhiễu tối thiểu mà nó chịu được là bao nhiêu?
3. Cài đặt ToT beam-search (giữ top-k ở mỗi tầng) và so sánh với BFS. Cách nào tốt hơn khi ngân sách token eo hẹp?
4. Đọc Mục 5.1 của LATS. Tái tạo số lượng quỹ đạo trên HumanEval: cần bao nhiêu rollout để đạt pass@1 như báo cáo?
5. Đọc phần thảo luận "when LATS helps less" trong bài báo LATS. Viết một đoạn quy tắc quyết định ánh xạ hình dạng bài toán sang chiến lược tìm kiếm.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Tree of Thoughts | "CoT phân nhánh" | Yao et al. — các nút tree of thought kèm tự đánh giá |
| LATS | "MCTS cho LLM" | Zhou et al. — hợp nhất ToT + ReAct + Reflexion dưới MCTS |
| UCT | "Upper confidence bound" | Công thức chọn cân bằng khai thác (Q) và khám phá (ln N / n) |
| Value function | "Trạng thái này tốt đến đâu" | Điểm do LLM sinh qua prompt hoặc phần thưởng môi trường; nuôi backprop |
| Policy | "Bộ đề xuất hành động" | Bộ sinh kiểu ReAct; phát ra thought/hành động kế tiếp ứng viên |
| Rollout | "Quỹ đạo mô phỏng" | Đi từ một nút tới lá bằng policy, chấm điểm bằng hàm giá trị |
| Backpropagate | "Cập nhật tổ tiên" | Đẩy phần thưởng của lá ngược lên đường đi, cập nhật số lần thăm và Q |
| Search cost | "Bùng nổ token" | Gấp 100-1000 lần CoT trên Game of 24; tính ngân sách trước khi áp dụng |

## Đọc thêm

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — bài báo kinh điển
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS với phản hồi Reflexion
- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — các mẫu subgraph cho tìm kiếm
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — tìm kiếm tiến hoá với bộ đánh giá bằng chương trình

---

[← Bài trước: Reflexion: Học tăng cường bằng lời](../../03-reflexion-verbal-rl/docs/vi.md) · [Bài sau: Self-Refine và CRITIC: cải thiện đầu ra theo vòng lặp →](../../05-self-refine-and-critic/docs/vi.md)
