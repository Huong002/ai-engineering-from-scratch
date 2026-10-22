# Đầu ra có cấu trúc & Giải mã có ràng buộc

> Yêu cầu LLM trả về JSON. Phần lớn thời gian bạn sẽ nhận được JSON. Trong production, chính chữ "phần lớn" là vấn đề. Giải mã có ràng buộc (constrained decoding) biến "phần lớn" thành "luôn luôn" bằng cách sửa logits trước khi lấy mẫu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 17 (Chatbot), Giai đoạn 5, Bài 19 (Token hoá subword)
**Time:** ~60 phút

## Vấn đề

Một bộ phân loại nhắc LLM: "Trả về một trong {positive, negative, neutral}." Mô hình trả về "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...". Parser của bạn crash. F1 của bộ phân loại là 0.0.

Sinh văn bản tự do không phải là một hợp đồng. Nó chỉ là một gợi ý. Một hệ thống production cần một hợp đồng.

Năm 2026 có ba lớp.

1. **Prompt.** Yêu cầu một cách lịch sự. "Chỉ trả về đối tượng JSON." Hiệu quả ~80% trên các mô hình frontier, thấp hơn trên mô hình nhỏ.
2. **API structured output gốc của nhà cung cấp.** OpenAI `response_format`, tool use của Anthropic, JSON mode của Gemini. Đáng tin cậy trên các schema được hỗ trợ. Khoá cứng vào nhà cung cấp.
3. **Giải mã có ràng buộc.** Sửa logits ở mỗi bước sinh để mô hình *không thể* phát ra token không hợp lệ. Hợp lệ 100% ngay từ cấu trúc. Chạy được trên mọi mô hình local.

Bài này xây trực giác cho cả ba lớp và chỉ ra khi nào nên dùng cái nào.

## Khái niệm

![Giải mã có ràng buộc che các token không hợp lệ ở mỗi bước](../assets/constrained-decoding.svg)

**Cách giải mã có ràng buộc hoạt động.** Ở mỗi bước sinh, LLM tạo ra một vector logit trên toàn bộ từ vựng (~100k token). Một *bộ xử lý logit* nằm giữa mô hình và bộ lấy mẫu. Nó tính xem token nào hợp lệ với vị trí hiện tại trong văn phạm (grammar) đích — JSON Schema, regex, văn phạm phi ngữ cảnh — rồi đặt logit của mọi token không hợp lệ về âm vô cực. Softmax trên phần logit còn lại chỉ dồn khối xác suất vào những phần tiếp nối hợp lệ.

Các hiện thực năm 2026:

- **Outlines.** Biên dịch JSON Schema hoặc regex thành một máy trạng thái hữu hạn (finite state machine). Mỗi token có tra cứu token-kế-tiếp-hợp-lệ O(1). Dựa trên FSM, nên schema đệ quy cần được làm phẳng.
- **XGrammar / llguidance.** Các engine văn phạm phi ngữ cảnh. Xử lý được JSON Schema đệ quy. Chi phí giải mã gần như bằng không. OpenAI đã ghi công llguidance trong hiện thực structured output năm 2025 của họ.
- **vLLM guided decoding.** Có sẵn `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` qua backend Outlines, XGrammar hoặc lm-format-enforcer.
- **Instructor.** Lớp bọc dựa trên Pydantic cho bất kỳ LLM nào. Thử lại khi validation thất bại. Đa nhà cung cấp, nhưng không sửa logits — nó dựa vào retry cộng với prompt có nhận biết structured output.

### Kết quả phản trực giác

Giải mã có ràng buộc thường *nhanh hơn* sinh không ràng buộc. Hai lý do. Thứ nhất, nó thu hẹp không gian tìm kiếm token kế tiếp. Thứ hai, các hiện thực khéo léo bỏ qua hẳn việc sinh token với những token bị ép buộc (phần khung như `{"name": "` — mọi byte đã được quyết định).

### Cái bẫy khiến bạn trả giá

Thứ tự trường có ý nghĩa. Đặt `answer` trước `reasoning`, và mô hình chốt câu trả lời trước khi nó suy nghĩ. JSON hợp lệ. Câu trả lời sai. Không có validation nào bắt được.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Thứ tự trường trong schema là logic, không phải định dạng.

```figure
constrained-decoder
```

## Tự tay làm

### Bước 1: sinh có ràng buộc bằng regex từ đầu

Xem `code/main.py` để có một hiện thực FSM độc lập. Ý tưởng cốt lõi trong 30 dòng:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM theo dõi xem tới thời điểm này ta đã thoả mãn những phần nào của văn phạm. `valid_tokens(state, tokenizer)` tính ra những token trong từ vựng có thể đẩy FSM tiến lên mà không rời khỏi một đường dẫn còn chấp nhận được — nghĩa là vẫn giữ một tiền tố hợp lệ (valid prefix).

### Bước 2: Outlines cho JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Không còn lỗi validation nào. Mãi mãi. FSM khiến đầu ra không hợp lệ trở nên không thể chạm tới.

### Bước 3: Instructor cho Pydantic độc lập nhà cung cấp

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Cơ chế khác. Instructor không đụng tới logits. Nó định dạng schema vào prompt, parse đầu ra, và thử lại khi validation thất bại (mặc định 3 lần). Chạy được với mọi nhà cung cấp. Retry làm tăng độ trễ và chi phí. Khả năng chuyển đổi giữa các nhà cung cấp chính là điểm bán.

### Bước 4: API gốc của nhà cung cấp

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Giải mã có ràng buộc phía server. Độ tin cậy ngang Outlines với các schema được hỗ trợ. Không phải quản lý mô hình local. Khoá bạn vào nhà cung cấp.

## Cạm bẫy

- **Schema đệ quy.** Outlines làm phẳng đệ quy tới một độ sâu cố định. Đầu ra dạng cây (bình luận lồng nhau, AST) cần XGrammar hoặc llguidance (dựa trên CFG).
- **Enum khổng lồ.** Enum 10.000 lựa chọn biên dịch chậm hoặc timeout. Chuyển sang retriever: dự đoán top-k ứng viên trước, rồi ràng buộc vào tập đó.
- **Văn phạm quá chặt.** Ép regex `date: "YYYY-MM-DD"` và mô hình không thể xuất `"unknown"` cho ngày bị thiếu. Mô hình bù bằng cách bịa ra một ngày. Hãy cho phép `null` hoặc một giá trị sentinel.
- **Chốt sớm.** Xem cái bẫy thứ tự trường ở trên. Luôn đặt reasoning trước.
- **JSON mode của nhà cung cấp mà không có schema.** JSON mode thuần chỉ bảo đảm JSON hợp lệ, không bảo đảm hợp lệ *với ca dùng của bạn*. Luôn cung cấp schema đầy đủ.

## Áp dụng

Stack của năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Mô hình OpenAI/Anthropic/Google, schema đơn giản | Structured output gốc của nhà cung cấp |
| Nhà cung cấp bất kỳ, quy trình Pydantic, chịu được retry | Instructor |
| Mô hình local, cần hợp lệ 100%, schema phẳng | Outlines (FSM) |
| Mô hình local, schema đệ quy | XGrammar hoặc llguidance |
| Tự vận hành server suy luận | vLLM guided decoding |
| Xử lý batch, chấp nhận retry | Instructor + mô hình rẻ nhất |

## Đóng gói

Lưu thành `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Bài tập

1. **Dễ.** Prompt một mô hình open-weights nhỏ (ví dụ Llama-3.2-3B) mà không có giải mã có ràng buộc cho `Review(sentiment, confidence, evidence_span)`. Đo tỉ lệ parse được thành JSON hợp lệ trên 100 review.
2. **Trung bình.** Cùng corpus đó với JSON mode của Outlines. So sánh tỉ lệ tuân thủ, độ trễ và độ chính xác ngữ nghĩa.
3. **Khó.** Tự hiện thực một bộ giải mã có ràng buộc bằng regex cho số điện thoại (`\d{3}-\d{3}-\d{4}`). Kiểm chứng 0 đầu ra không hợp lệ trên 1000 mẫu.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Constrained decoding | Ép đầu ra hợp lệ | Che logit của token không hợp lệ ở mỗi bước sinh. |
| Logit processor | Thứ ràng buộc đầu ra | Hàm: `(logits, state) -> masked_logits`. |
| FSM | Máy trạng thái hữu hạn | Biểu diễn văn phạm đã biên dịch; tra cứu token-kế-tiếp-hợp-lệ O(1). |
| CFG | Văn phạm phi ngữ cảnh | Văn phạm xử lý được đệ quy; chậm hơn nhưng biểu đạt mạnh hơn FSM. |
| Schema field order | Có quan trọng không? | Có — trường đầu tiên là trường chốt; luôn đặt reasoning trước answer. |
| Guided decoding | Tên mà vLLM gọi nó | Cùng khái niệm, tích hợp vào server suy luận. |
| JSON mode | Phiên bản đầu của OpenAI | Bảo đảm cú pháp JSON; KHÔNG bảo đảm khớp schema. |

## Đọc thêm

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — bài báo Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — giải mã có ràng buộc dựa trên CFG, nhanh.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — tích hợp trong server suy luận.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — tham chiếu API cùng các cạm bẫy.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + retry trên nhiều nhà cung cấp.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — benchmark 6 framework giải mã có ràng buộc.

---

[← Bài trước: Token hoá dưới từ](../../19-subword-tokenization/docs/vi.md) · [Bài sau: Natural Language Inference (suy luận ngôn ngữ tự nhiên) →](../../21-nli-textual-entailment/docs/vi.md)
