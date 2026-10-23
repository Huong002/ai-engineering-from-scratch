# Reflexion: Học tăng cường bằng lời

> RL dựa trên gradient cần hàng nghìn trial và một cụm GPU để sửa một kiểu lỗi. Reflexion (tự phản tư bằng lời; Shinn et al., NeurIPS 2023) làm việc đó bằng ngôn ngữ tự nhiên: sau mỗi trial thất bại, agent viết một lời phản tư, lưu nó vào episodic memory (bộ nhớ theo tình huống), rồi lấy bộ nhớ đó làm điều kiện cho trial tiếp theo. Đây là khuôn mẫu đằng sau sleep-time compute của Letta, các learnings trong CLAUDE.md của Claude Code, và learn-rule của pro-workflow.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 02 (ReWOO)
**Time:** ~60 phút

## Mục tiêu học tập

- Kể tên ba thành phần của Reflexion (Actor, Evaluator, Self-Reflector) và vai trò của episodic memory.
- Cài đặt một vòng lặp Reflexion bằng stdlib với evaluator nhị phân, buffer phản tư và các lần thử lại hoàn toàn mới.
- Chọn giữa các nguồn phản hồi scalar, heuristic và self-evaluated cho một bài toán cụ thể.
- Giải thích vì sao verbal reinforcement learning (học tăng cường bằng lời) bắt được những lỗi mà RL dựa trên gradient phải mất hàng nghìn trial mới sửa xong.

## Vấn đề

Một agent làm hỏng một bài toán. Trong RL chuẩn, bạn sẽ phải chạy thêm hàng nghìn trial, tính gradient, cập nhật trọng số. Đắt, chậm, và hầu hết agent chạy production không có ngân sách huấn luyện cho mỗi lần thất bại.

Reflexion (Shinn et al., arXiv:2303.11366) đặt một câu hỏi khác: nếu agent chỉ nghĩ xem vì sao mình thất bại rồi thử lại với suy nghĩ đó nằm trong prompt thì sao? Không cập nhật trọng số. Không gradient. Chỉ là ngôn ngữ tự nhiên được lưu lại giữa các trial.

Kết quả: trên ALFWorld nó vượt ReAct và các baseline khác không fine-tune. Trên HotpotQA nó cải thiện so với ReAct. Trên sinh code (HumanEval/MBPP) nó đạt state of the art vào thời điểm đó. Tất cả mà không cần một bước gradient nào.

## Khái niệm

### Ba thành phần

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

Thêm một cấu trúc dữ liệu:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

Một trial chạy Actor. Evaluator chấm điểm. Nếu điểm thấp, Self-Reflector — thành phần phụ trách self-reflection (tự phản tư) — tạo ra một lời phản tư ("Tôi chọn sai công cụ vì tôi đọc nhầm câu hỏi, tưởng nó hỏi về X trong khi nó hỏi về Y"). Lời phản tư đi vào episodic memory. Trial tiếp theo bắt đầu lại từ đầu nhưng thấy được lời phản tư đó.

### Ba loại evaluator

1. **Scalar** — một tín hiệu nhị phân từ bên ngoài. ALFWorld thành công hay thất bại. Test HumanEval đạt hay không đạt. Đơn giản nhất, tín hiệu mạnh nhất.
2. **Heuristic** — các dấu hiệu thất bại định trước. "Nếu agent tạo ra cùng một hành động hai lần liên tiếp, đánh dấu là kẹt." "Nếu trajectory vượt quá 50 bước, đánh dấu là kém hiệu quả."
3. **Self-evaluated** — LLM tự chấm điểm trajectory của chính nó. Cần khi không có ground truth. Tín hiệu yếu hơn; ghép tốt với kiểm chứng dựa trên công cụ (Bài 05 — CRITIC).

Mặc định của năm 2026 là pha trộn: scalar khi có, self-eval khi không, heuristic làm lan can an toàn.

### Vì sao cách này tổng quát hóa được

Reflexion không hẳn là một thuật toán mới mà là một khuôn mẫu đã được đặt tên. Gần như mọi agent "tự chữa lành" trong production đều chạy một biến thể nào đó:

- Sleep-time compute của Letta (Bài 08): một agent riêng phản tư về các cuộc hội thoại trước và ghi vào các memory block.
- Khuôn mẫu `CLAUDE.md` / "save memory" của Claude Code: các phản tư được ghi lại thành learnings, đặt trước cho những phiên sau.
- Lệnh `/learn-rule` của pro-workflow: các chỉnh sửa được ghi lại thành quy tắc rõ ràng.
- Các reflection node của LangGraph: một node chấm điểm output và chuyển sang bước tinh chỉnh nếu cần.

Tất cả đều xuất phát từ cùng một nhận định: ngôn ngữ tự nhiên là một môi trường đủ giàu để mang "điều tôi học được từ thất bại" qua các lần chạy.

### Khi nào hiệu quả, khi nào không

Reflexion hiệu quả khi:

- Có tín hiệu thất bại rõ ràng (test hỏng, công cụ lỗi, đáp án sai).
- Lớp bài toán có thể tái lập (cùng kiểu câu hỏi có thể được hỏi lại).
- Lời phản tư còn chỗ để cải thiện trajectory (đủ ngân sách hành động).

Reflexion không giúp được khi:

- Agent đã thành công ngay lần đầu.
- Thất bại đến từ bên ngoài (mạng sập, công cụ hỏng) — phản tư về "mạng đã sập" không giúp gì cho những lần chạy sau.
- Lời phản tư biến thành mê tín — lưu lại một câu chuyện về một lần chạy chập chờn cá biệt.

Cạm bẫy năm 2026: memory rot. Các phản tư tích tụ; một số đã lỗi thời hoặc sai; các lần chạy lại chậm dần khi buffer episodic phình ra. Cách giảm thiểu: nén định kỳ (Bài 06), TTL cho phản tư, hoặc một agent dọn dẹp lúc rảnh riêng (Letta).

```figure
react-trace
```

## Tự tay làm

`code/main.py` cài đặt Reflexion trên một bài đố nhỏ: tạo một danh sách 3 phần tử có tổng bằng giá trị đích. Actor phát ra các danh sách ứng viên; Evaluator kiểm tra tổng; Self-Reflector viết một dòng về chỗ sai. Lời phản tư đi vào episodic memory cho trial tiếp theo.

Các thành phần:

- `Actor` — một policy viết cứng, tự cải thiện khi thấy các phản tư.
- `Evaluator.binary()` — đạt/không đạt theo tổng đích.
- `SelfReflector` — sinh một dòng chẩn đoán thất bại.
- `EpisodicMemory` — một danh sách có giới hạn với ngữ nghĩa TTL.

Chạy nó:

```
python3 code/main.py
```

Trace cho thấy ba trial. Trial 1 thất bại, một lời phản tư được lưu, trial 2 thấy lời phản tư và tiến bộ nhưng vẫn thất bại, trial 3 thành công. So sánh với một lần chạy baseline (không có phản tư) — nó kẹt mãi ở đáp án của trial 1.

## Áp dụng

LangGraph cung cấp reflection như một node pattern. Lệnh `/memory` của Claude Code và `/learn-rule` của pro-workflow đưa buffer episodic ra ngoài thành một tệp markdown. Sleep-time compute của Letta chạy Self-Reflector vào lúc rảnh để agent chính vẫn bị ràng buộc bởi độ trễ. OpenAI Agents SDK không có sẵn Reflexion; bạn tự dựng nó bằng một Guardrail tuỳ chỉnh loại bỏ trajectory theo điểm và một `Session` bộ nhớ tồn tại qua các lần chạy.

## Đóng gói

`outputs/skill-reflexion-buffer.md` tạo và duy trì một buffer episodic với việc bắt phản tư, TTL và khử trùng lặp. Cho một lớp bài toán và một thất bại, nó phát ra một lời phản tư thực sự giúp trial tiếp theo (không phải câu chung chung "cẩn thận hơn nhé").

## Bài tập

1. Đổi từ evaluator nhị phân sang scalar trả về một độ đo khoảng cách (cách đích bao xa). Nó hội tụ nhanh hơn không?
2. Thêm TTL 10 trial cho các phản tư. Sau mốc đó, những phản tư cũ gây hại hay có ích?
3. Cài đặt evaluator heuristic: đánh dấu trial là kẹt nếu cùng một hành động lặp lại. Điều này tương tác thế nào với Self-Reflector?
4. Chạy Reflexion với một Actor đối nghịch, phớt lờ các phản tư. Cần tối thiểu bao nhiêu kỹ thuật viết prompt cho phản tư để buộc Actor phải để ý tới chúng?
5. Đọc Mục 4 của bài báo Reflexion về AlfWorld. Tái hiện về mặt khái niệm mức cải thiện 130% tỉ lệ thành công: khác biệt then chốt so với ReAct nguyên bản là gì?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Reflexion | "Tự sửa lỗi" | Shinn et al. 2023 — Actor, Evaluator, Self-Reflector cùng episodic memory |
| Verbal reinforcement | "Học mà không cần gradient" | Lời phản tư bằng ngôn ngữ tự nhiên được đặt trước prompt của trial tiếp theo |
| Episodic memory | "Phản tư theo từng bài" | Buffer có giới hạn chứa các phản tư trước đó cho một lớp bài toán |
| Scalar evaluator | "Tín hiệu thành công nhị phân" | Đạt/không đạt hoặc điểm số từ ground truth |
| Heuristic evaluator | "Bộ phát hiện dựa trên mẫu" | Các dấu hiệu thất bại định trước (ví dụ: lặp vô hạn, quá nhiều bước) |
| Self-evaluator | "LLM làm giám khảo trên trace của chính nó" | Phương án dự phòng tín hiệu yếu khi không có ground truth — ghép với kiểm chứng dựa trên công cụ |
| Memory rot | "Phản tư cũ mòn" | Buffer episodic đầy các mục lỗi thời; sửa bằng nén định kỳ/TTL |
| Sleep-time reflection | "Tự phản tư bất đồng bộ" | Chạy Self-Reflector ngoài đường nóng để agent chính giữ độ trễ thấp |

## Đọc thêm

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — bài báo gốc
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — phản tư bất đồng bộ trong production
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — quản lý buffer episodic như một phần của context
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — khuôn mẫu reflection node
