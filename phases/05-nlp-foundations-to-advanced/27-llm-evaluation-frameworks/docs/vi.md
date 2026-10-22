# Đánh giá LLM — RAGAS, DeepEval, G-Eval

> Exact-match và F1 bỏ sót tương đương về ngữ nghĩa. Chấm bởi người thì không mở rộng được. LLM-as-judge (LLM làm giám khảo) là câu trả lời cho môi trường production — với đủ hiệu chuẩn để tin được con số.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 13 (Hỏi đáp), Giai đoạn 5, Bài 14 (Truy hồi thông tin)
**Time:** ~75 phút

## Vấn đề

Hệ thống RAG của bạn trả lời: "June 29th, 2007."
Câu tham chiếu chuẩn là: "June 29, 2007."
Exact Match cho 0 điểm. F1 được khoảng 75%. Người chấm sẽ cho 100%.

Giờ nhân lên 10.000 test case. Nhân tiếp với mỗi thay đổi ở retriever, chunking, prompt hay model. Bạn cần một evaluator hiểu ngữ nghĩa, chạy rẻ ở quy mô lớn, không nói dối về suy giảm chất lượng (regression), và chỉ ra đúng những kiểu hỏng.

Năm 2026 có ba framework giải quyết bài toán này.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Bốn metric cho RAG (faithfulness, answer-relevance, context-precision, context-recall) với backend NLI + LLM-judge. Có nghiên cứu hậu thuẫn, gọn nhẹ.
- **DeepEval.** Pytest dành cho LLM. G-Eval, task-completion, hallucination, bias metric. Sinh ra cho CI/CD.
- **G-Eval.** Một phương pháp (và cũng là một metric của DeepEval): LLM-as-judge với chain-of-thought, tiêu chí tùy chỉnh, điểm 0-1.

Cả ba đều dựa vào LLM-as-judge. Bài này xây trực giác về phương pháp và lớp tin cậy bao quanh nó.

## Khái niệm

![Bốn chiều đánh giá, kiến trúc LLM-as-judge](../assets/llm-evaluation.svg)

**LLM-as-judge.** Thay một metric tĩnh bằng một LLM chấm đầu ra theo rubric (thang điểm chấm). Cho `(query, context, answer)`, nhắc một LLM giám khảo: "Chấm 0-1 theo faithfulness." Trả về điểm.

Vì sao nó hiệu quả: LLM xấp xỉ được đánh giá của con người với chi phí chỉ bằng một phần rất nhỏ. GPT-4o-mini khoảng $0.003 mỗi case được chấm, đủ để chạy eval hồi quy 1000 mẫu dưới $5.

Vì sao nó hỏng mà không ai biết:

1. **Thiên lệch của giám khảo.** Giám khảo thích câu trả lời dài hơn, câu trả lời từ chính họ model, và câu trả lời khớp phong cách của prompt.
2. **Parse JSON thất bại.** JSON hỏng → điểm NaN → bị loại khỏi tổng hợp mà không báo gì. Người dùng RAGAS biết rõ nỗi đau này. Hãy chặn bằng try/except + một chế độ thất bại tường minh.
3. **Trôi theo phiên bản model.** Nâng cấp giám khảo làm đổi mọi metric. Hãy ghim model giám khảo + phiên bản.

**Bốn metric của RAG.**

| Metric | Câu hỏi | Backend |
|--------|----------|---------|
| Faithfulness | Mỗi khẳng định trong câu trả lời có đến từ ngữ cảnh truy hồi được không? | Suy diễn dựa trên NLI |
| Answer relevance | Câu trả lời có giải quyết đúng câu hỏi không? | Sinh câu hỏi giả định từ câu trả lời; so sánh với câu hỏi thật |
| Context precision | Trong các chunk truy hồi được, tỉ lệ liên quan là bao nhiêu? | LLM-judge |
| Context recall | Truy hồi có lấy về đủ mọi thứ cần thiết không? | LLM-judge so với đáp án chuẩn |

**G-Eval.** Định nghĩa một tiêu chí riêng: "Câu trả lời có trích đúng nguồn không?" Framework tự mở rộng thành các bước đánh giá chain-of-thought, rồi chấm 0-1. Hợp cho những chiều chất lượng đặc thù miền mà RAGAS không phủ.

**Calibration (hiệu chuẩn).** Đừng bao giờ tin điểm thô của giám khảo trước khi có tương quan với nhãn của người. Chạy 100 ví dụ gán nhãn tay. Vẽ giám khảo so với người. Tính Spearman rho. Nếu rho < 0.7, rubric của giám khảo cần làm lại.

```figure
n5-judge-gauge
```

## Tự tay làm

### Bước 1: faithfulness với NLI (kiểu RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Tách câu trả lời thành các khẳng định nguyên tử. Kiểm tra từng khẳng định với ngữ cảnh đã truy hồi bằng NLI. Faithfulness = tỉ lệ khẳng định được ngữ cảnh chống đỡ.

### Bước 2: độ liên quan của câu trả lời

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Nếu câu trả lời ngụ ý những câu hỏi khác với câu được hỏi, độ liên quan sẽ tụt.

### Bước 3: metric tùy chỉnh với G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Các bước đánh giá (evaluation steps) chính là rubric. Bước tường minh ổn định hơn prompt "chấm 0-1" ngầm hiểu.

### Bước 4: cổng CI

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Đóng gói thành một file pytest. Chạy trên mọi PR. Chặn merge khi có suy giảm.

### Bước 5: eval đồ chơi từ đầu

Xem `code/main.py`. Bản xấp xỉ chỉ dùng stdlib cho faithfulness (phần trùng lặp giữa khẳng định trong câu trả lời và ngữ cảnh) và relevance (phần trùng lặp giữa token câu trả lời và token câu hỏi). Không dùng cho production. Nhưng cho thấy hình dạng của bài toán.

## Cạm bẫy

- **Không hiệu chuẩn.** Một giám khảo có tương quan 0.3 với nhãn người chỉ là nhiễu. Bắt buộc có một lượt hiệu chuẩn trước khi đưa vào dùng.
- **Tự chấm chính mình.** Dùng cùng một LLM để sinh và để chấm làm điểm phồng thêm 10-20%. Hãy dùng giám khảo thuộc họ model khác.
- **Thiên lệch vị trí khi chấm pairwise comparison (so sánh cặp).** Giám khảo thích phương án được đưa ra trước. Luôn xáo trộn thứ tự và chạy cả hai chiều.
- **Điểm tổng hợp che mất lỗi.** Điểm trung bình 0.85 thường che 5% lỗi thảm hoạ. Luôn soi nhóm điểm thấp nhất.
- **Bộ dữ liệu vàng bị mục.** Tập eval không được version hoá, trôi dần theo thời gian, làm hỏng so sánh dọc. Gắn tag cho bộ dữ liệu mỗi lần thay đổi.
- **Chi phí LLM.** Ở quy mô lớn, các lượt gọi giám khảo chiếm phần lớn chi phí. Dùng model rẻ nhất mà vẫn đạt ngưỡng hiệu chuẩn. GPT-4o-mini, Claude Haiku, Mistral-small.

## Áp dụng

Bộ stack năm 2026:

| Trường hợp dùng | Framework |
|---------|-----------|
| Giám sát chất lượng RAG | RAGAS (4 metric) |
| Cổng chống suy giảm trong CI/CD | DeepEval + pytest |
| Tiêu chí riêng theo miền | G-Eval trong DeepEval |
| Giám sát lưu lượng trực tiếp | RAGAS ở chế độ không cần tham chiếu |
| Kiểm tra mẫu có người trong vòng lặp | LangSmith hoặc Phoenix với giao diện gán nhãn |
| Red-teaming / đánh giá an toàn | Promptfoo + DeepEval |

Stack điển hình: RAGAS để giám sát, DeepEval cho CI, G-Eval cho những chiều mới. Chạy cả ba; chúng bất đồng một cách hữu ích.

## Đóng gói

Lưu thành `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Bài tập

1. **Dễ.** Dùng RAGAS trên 10 ví dụ RAG có hallucination đã biết. Kiểm tra rằng metric faithfulness bắt được từng ví dụ.
2. **Trung bình.** Gán nhãn tay 50 câu trả lời QA theo thang 0-1 về độ đúng. Chấm bằng G-Eval. Đo Spearman rho giữa giám khảo và người.
3. **Khó.** Dựng một cổng CI pytest với DeepEval. Cố tình làm retriever suy giảm. Kiểm tra rằng cổng fail. Thêm cảnh báo nhóm điểm thấp nhất bằng cách kiểm tra ngưỡng trên 10% thấp nhất.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| LLM-as-judge | Chấm điểm bằng LLM | Nhắc một model giám khảo chấm đầu ra 0-1 theo rubric. |
| RAGAS | Thư viện metric cho RAG | Framework eval mã nguồn mở với 4 metric RAG không cần tham chiếu. |
| Faithfulness | Câu trả lời có bám nguồn không? | Tỉ lệ khẳng định trong câu trả lời được ngữ cảnh truy hồi chống đỡ. |
| Context precision | Các chunk truy hồi có liên quan không? | Tỉ lệ chunk trong top-K thực sự hữu ích. |
| Context recall | Truy hồi có tìm đủ không? | Tỉ lệ khẳng định trong đáp án chuẩn được các chunk truy hồi chống đỡ. |
| G-Eval | Giám khảo LLM tùy chỉnh | Rubric + các bước đánh giá chain-of-thought + điểm 0-1. |
| Calibration | Tin nhưng phải kiểm | Tương quan Spearman giữa điểm giám khảo và điểm người chấm. |

## Đọc thêm

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — bài báo RAGAS.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — bài báo G-Eval.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — stack production mở.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — thiên lệch, hiệu chuẩn, giới hạn.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — framework hợp nhất tích hợp RAGAS, DeepEval, Phoenix.

---

[← Bài trước: Trích xuất quan hệ & Xây dựng đồ thị tri thức](../../26-relation-extraction-kg/docs/vi.md) · [Bài sau: Đánh giá ngữ cảnh dài →](../../28-long-context-evaluation/docs/vi.md)
