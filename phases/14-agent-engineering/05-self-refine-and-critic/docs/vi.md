# Self-Refine và CRITIC: cải thiện đầu ra theo vòng lặp

> Self-Refine (tự tinh chỉnh, Madaan et al., 2023) dùng một LLM với ba vai — tạo, phản hồi, tinh chỉnh — trong một vòng phản hồi (feedback loop). Mức tăng trung bình: +20 tuyệt đối trên 7 tác vụ. CRITIC (Gou et al., 2023) làm bước phản hồi vững hơn bằng cách đưa việc kiểm chứng qua các công cụ bên ngoài. Năm 2026, mẫu hình này có mặt trong mọi framework dưới tên "evaluator-optimizer" (Anthropic) hoặc một vòng guardrail (OpenAI Agents SDK).

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 03 (Reflexion)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu ba prompt của Self-Refine (tạo, phản hồi, tinh chỉnh) và giải thích vì sao lịch sử lại quan trọng với prompt tinh chỉnh.
- Giải thích điểm mấu chốt của CRITIC: LLM không đáng tin khi tự kiểm chứng nếu không có căn cứ bên ngoài.
- Cài đặt một vòng Self-Refine bằng stdlib, có lịch sử và tuỳ chọn verifier bên ngoài.
- Ánh xạ mẫu hình này sang workflow "evaluator-optimizer" của Anthropic và output guardrail của OpenAI Agents SDK.

## Vấn đề

Một agent đưa ra câu trả lời gần đúng. Có thể một dòng code bị lỗi cú pháp. Có thể bản tóm tắt quá dài. Có thể kế hoạch bỏ sót một trường hợp biên. Điều bạn muốn là: agent tự phê bình đầu ra của nó, rồi sửa lại.

Self-Refine cho thấy điều này khả thi với một mô hình duy nhất, không cần dữ liệu huấn luyện, không cần RL. Nhưng có một cái bẫy: LLM tự kiểm chứng kém với các sự kiện khó. CRITIC chỉ ra cách khắc phục — đưa bước verify qua các công cụ bên ngoài (tìm kiếm, trình thông dịch code, máy tính, trình chạy test).

Gộp lại, hai bài báo này định hình mặc định năm 2026 cho việc cải thiện lặp: tạo, kiểm chứng (bên ngoài khi có thể), tinh chỉnh, dừng khi verifier chấm đạt.

## Khái niệm

### Self-Refine (Madaan et al., NeurIPS 2023)

Một LLM, ba vai:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

Chi tiết then chốt: `refine` thấy toàn bộ lịch sử — mọi đầu ra và lời phê trước đó — nên nó không lặp lại sai lầm. Bài báo có ablation phần này: bỏ lịch sử là chất lượng tụt rõ rệt.

Điểm nhấn: cải thiện +20 tuyệt đối, trung bình trên 7 tác vụ (toán, code, từ viết tắt, hội thoại), tính cả GPT-4. Không huấn luyện, không công cụ ngoài, một mô hình duy nhất.

### CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024)

Điểm yếu của Self-Refine: bước phản hồi là một LLM tự cho điểm chính nó. Với các tuyên bố về sự kiện, cách này không đáng tin (một ảo giác thường trông rất thuyết phục với chính mô hình tạo ra nó). CRITIC thay `feedback(task, output)` bằng `verify(task, output, tools)`, trong đó `tools` gồm:

- Một công cụ tìm kiếm cho các tuyên bố về sự kiện.
- Một trình thông dịch code để kiểm tra tính đúng của code.
- Một máy tính cho phép tính số học.
- Các verifier theo miền (unit test, trình kiểm tra kiểu, linter).

Verifier tạo ra một lời phê có cấu trúc, dựa trên kết quả công cụ. Bộ tinh chỉnh sau đó dựa vào lời phê này.

Điểm nhấn: CRITIC vượt Self-Refine trên các tác vụ sự kiện vì lời phê có căn cứ. Với tác vụ không có verifier bên ngoài (viết sáng tạo, định dạng), CRITIC thu gọn thành Self-Refine.

### Điều kiện dừng

Hai dạng phổ biến:

1. **Verifier chấm đạt.** Test bên ngoài trả về thành công. Ưu tiên khi có (unit test, trình kiểm tra kiểu, assert của guardrail).
2. **Không có phản hồi nào.** Mô hình nói "đầu ra ổn rồi". Rẻ hơn nhưng không đáng tin; hãy ghép với một trần số vòng lặp.

Mặc định 2026: kết hợp cả hai. "Dừng nếu verifier chấm đạt HOẶC mô hình nói ổn VÀ số vòng >= 2 HOẶC số vòng >= max_iterations."

### Evaluator-Optimizer (Anthropic, 2024)

Bài viết tháng 12/2024 của Anthropic gọi đây là một trong năm mẫu workflow. Hai vai:

- Evaluator: chấm điểm đầu ra và tạo lời phê.
- Optimizer: sửa đầu ra dựa trên lời phê.

Lặp cho tới khi evaluator chấm đạt. Đây chính là Self-Refine/CRITIC theo cách gọi của Anthropic. Chi tiết kỹ thuật quan trọng mà Anthropic bổ sung: prompt của evaluator và optimizer phải khác nhau đáng kể để mô hình không chỉ đóng dấu "đạt".

### Output guardrail của OpenAI Agents SDK

OpenAI Agents SDK mang mẫu hình này vào dưới tên "output guardrails". Guardrail là một validator chạy trên đầu ra cuối cùng của agent. Nếu guardrail bắt lỗi (ném `OutputGuardrailTripwireTriggered`), đầu ra bị từ chối và agent có thể thử lại. Guardrail có thể gọi công cụ (kiểu CRITIC) hoặc là hàm thuần (kiểu Self-Refine).

### Những cái bẫy năm 2026

- **Vòng đóng dấu "đạt".** Cùng một mô hình vừa tạo vừa phê với cùng kiểu prompt sẽ hội tụ về "trông ổn đấy". Hãy dùng prompt khác nhau về cấu trúc, hoặc một mô hình nhỏ, rẻ hơn để phê bình.
- **Tinh chỉnh quá đà.** Mỗi vòng tinh chỉnh thêm độ trễ và token. Dự trù 1-3 vòng; quá đó thì chuyển sang người duyệt.
- **CRITIC cho tác vụ tầm thường.** Nếu không có verifier bên ngoài, CRITIC thoái hoá thành Self-Refine; đừng trả giá độ trễ cho một verifier rỗng.

```figure
self-refine
```

## Tự tay làm

`code/main.py` cài đặt Self-Refine và CRITIC trên một tác vụ đồ chơi: tạo một danh sách gạch đầu dòng ngắn từ một chủ đề. Verifier kiểm tra định dạng (3 gạch đầu dòng, mỗi dòng dưới 60 ký tự). CRITIC thêm một "fact verifier" bên ngoài, phạt các ảo giác đã biết.

Các thành phần:

- `generate` — bộ tạo theo kịch bản.
- `feedback` — tự phê kiểu LLM.
- `verify_external` — verifier có căn cứ kiểu CRITIC.
- `refine` — viết lại đầu ra dựa trên lịch sử.
- Điều kiện dừng — verifier chấm đạt hoặc tối đa 4 vòng.

Chạy thử:

```
python3 code/main.py
```

So sánh lượt chạy Self-Refine và CRITIC. CRITIC bắt được một lỗi sự kiện mà Self-Refine bỏ sót, vì verifier bên ngoài có căn cứ mà critic (người phê bình) nội tại không có.

## Áp dụng

Evaluator-optimizer của Anthropic chính là mẫu hình này bằng ngôn ngữ thân thiện với Claude. Output guardrail của OpenAI Agents SDK có dạng CRITIC (guardrail gọi được công cụ). LangGraph có một reflection node đọc lên giống Self-Refine. Gemini 2.5 Computer Use của Google thêm một evaluator an toàn theo từng bước, một biến thể của CRITIC: mọi hành động đều được kiểm chứng trước khi thực thi.

## Đóng gói

`outputs/skill-refine-loop.md` cấu hình một vòng evaluator-optimizer dựa trên dạng tác vụ, mức sẵn có của verifier và ngân sách số vòng. Nó sinh ra prompt cho generator, evaluator/verifier và optimizer, kèm một chính sách dừng.

## Bài tập

1. Chạy bản đồ chơi với max_iterations=1. CRITIC có còn giúp được gì không?
2. Thay verifier bên ngoài bằng một bản nhiễu (30% dương tính giả ngẫu nhiên). Vòng lặp sẽ làm gì? Đây là thực tế năm 2026 của hầu hết các chồng guardrail.
3. Cài đặt biến thể "generator-critic trên các mô hình khác nhau": mô hình lớn tạo, mô hình nhỏ phê. Nó có thắng được cùng-mô-hình không?
4. Đọc Mục 3 của CRITIC (arXiv:2305.11738 v4). Kể tên ba nhóm công cụ kiểm chứng và nêu một ví dụ cho mỗi nhóm.
5. Ánh xạ `output_guardrails` của OpenAI Agents SDK sang vai verifier của CRITIC. SDK làm sai điều gì, và làm đúng điều gì?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Self-Refine | "LLM tự sửa mình" | Vòng tạo -> phản hồi -> tinh chỉnh trong một mô hình, có lịch sử |
| CRITIC | "Kiểm chứng có công cụ làm căn cứ" | Thay phản hồi bằng một verifier bên ngoài (tìm kiếm, code, máy tính, test) |
| Evaluator-Optimizer | "Mẫu workflow của Anthropic" | Hai vai — evaluator chấm điểm, optimizer sửa — lặp tới khi hội tụ |
| Output guardrail | "Kiểm tra sau khi xong" | Validator của OpenAI Agents SDK chạy sau khi agent tạo đầu ra |
| Verify step | "Pha phê bình" | Quyết định chịu lực: có căn cứ hay tự chấm |
| Refine history | "Những gì mô hình đã thử" | Các đầu ra + lời phê trước đó được thêm vào đầu prompt tinh chỉnh; bỏ đi là chất lượng sụp |
| Rubber-stamp loop | "Lỗi tự đồng ý" | Phê với cùng prompt trả về "trông ổn"; sửa bằng prompt khác cấu trúc |
| Stop condition | "Phép thử hội tụ" | Verifier chấm đạt HOẶC không có phản hồi VÀ có trần số vòng; đừng bao giờ chỉ một điều kiện |

## Đọc thêm

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — bài báo gốc
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — kiểm chứng có căn cứ từ công cụ
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — mẫu workflow evaluator-optimizer
- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — output guardrail như những verifier dạng CRITIC
