# Natural Language Inference (suy luận ngôn ngữ tự nhiên) — Kéo theo văn bản

> "t entails h" nghĩa là một người đọc `t` sẽ kết luận `h` đúng. NLI là bài toán dự đoán entailment (kéo theo) / contradiction (mâu thuẫn) / neutral (trung tính). Bề ngoài thì nhàm chán, nhưng trong production nó là thanh chịu lực.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 05 (Phân tích cảm xúc), Giai đoạn 5, Bài 13 (Hỏi đáp)
**Time:** ~60 phút

## Vấn đề

Bạn đã xây một công cụ tóm tắt. Nó tạo ra một bản tóm tắt. Làm sao bạn biết bản tóm tắt đó không chứa hallucination (ảo giác)?

Bạn đã xây một chatbot. Nó trả lời "có". Làm sao bạn biết câu trả lời đó được đoạn văn bản truy xuất hỗ trợ?

Bạn cần phân loại 10.000 bài báo theo chủ đề. Bạn không có nhãn huấn luyện. Liệu có thể tái sử dụng một mô hình?

Cả ba vấn đề đều quy về Natural Language Inference. NLI hỏi: cho một premise (tiền đề) `t` và hypothesis (giả thuyết) `h`, thì `h` có được `t` kéo theo, bị `t` mâu thuẫn, hay trung tính (không liên quan)?

- **Kiểm tra hallucination:** `t` = tài liệu nguồn, `h` = khẳng định trong bản tóm tắt. Không kéo theo = hallucination.
- **QA có căn cứ:** `t` = đoạn văn bản truy xuất, `h` = câu trả lời được sinh ra. Không kéo theo = bịa đặt.
- **Phân loại zero-shot (không cần huấn luyện):** `t` = tài liệu, `h` = nhãn được diễn đạt bằng lời ("This is about sports"). Kéo theo = nhãn được dự đoán.

Một bài toán, ba ứng dụng production. Đây là lý do mọi framework đánh giá RAG đều mang sẵn một mô hình NLI bên trong.

## Khái niệm

![NLI: phân loại ba nhánh, tiền đề so với giả thuyết](../assets/nli.svg)

**Ba nhãn.**

- **Entailment.** `t` → `h`. "The cat is on the mat" kéo theo "There is a cat."
- **Contradiction.** `t` → ¬`h`. "The cat is on the mat" mâu thuẫn với "There is no cat."
- **Neutral.** Không suy ra được gì theo cả hai hướng. "The cat is on the mat" là trung tính với "The cat is hungry."

**Không phải suy luận logic.** NLI là suy luận ngôn ngữ *tự nhiên* — điều mà một người đọc điển hình sẽ suy ra, không phải logic chặt chẽ. "John walked his dog" kéo theo "John has a dog" trong NLI, nhưng logic bậc một chặt chẽ chỉ chấp nhận điều đó nếu bạn tiên đề hoá quan hệ sở hữu.

**Bộ dữ liệu.**

- **SNLI** (2015). 570k cặp được con người gán nhãn (annotation), tiền đề là chú thích ảnh. Miền hẹp.
- **MultiNLI** (2017). 433k cặp trải trên 10 thể loại. Kho ngữ liệu huấn luyện chuẩn của năm 2026.
- **ANLI** (2019). Adversarial NLI. Con người viết những ví dụ được thiết kế riêng để đánh bại các mô hình hiện có. Khó hơn.
- **DocNLI, ConTRoL** (2020–21). Tiền đề dài bằng cả tài liệu. Kiểm tra suy luận nhiều bước và tầm xa.

**Kiến trúc.** Một encoder transformer (BERT, RoBERTa, DeBERTa) đọc `[CLS] premise [SEP] hypothesis [SEP]`. Biểu diễn `[CLS]` đi vào một softmax 3 lớp. Huấn luyện trên MNLI, đánh giá trên các benchmark giữ riêng, đạt độ chính xác 90%+ trên các cặp cùng phân phối.

**Zero-shot qua NLI.** Cho một tài liệu và các nhãn ứng viên, biến mỗi nhãn thành một giả thuyết ("This text is about sports"). Tính xác suất kéo theo cho từng nhãn. Chọn nhãn lớn nhất. Đây là cơ chế đằng sau pipeline `zero-shot-classification` của Hugging Face.

```figure
nli-router
```

## Tự tay làm

### Bước 1: chạy một mô hình NLI đã huấn luyện trước

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Với NLI trong production, `facebook/bart-large-mnli` và `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` là những lựa chọn mặc định mở. DeBERTa-v3 đứng đầu các bảng xếp hạng.

### Bước 2: phân loại zero-shot

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Template mặc định là "This example is about {label}." Tuỳ biến bằng `hypothesis_template`. Không cần dữ liệu huấn luyện. Không cần fine-tuning. Chạy được ngay.

### Bước 3: kiểm tra faithfulness cho RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Đây là lõi của RAGAS faithfulness. Tách câu trả lời được sinh thành các khẳng định nguyên tử. Kiểm tra từng khẳng định với ngữ cảnh truy xuất. Báo cáo tỉ lệ khẳng định được kéo theo.

### Bước 4: bộ phân loại NLI tự viết tay (mang tính khái niệm)

Xem `code/main.py` để thấy một món đồ chơi chỉ dùng stdlib: tiền đề và giả thuyết được so sánh qua trùng lặp từ vựng + phát hiện phủ định. Không cạnh tranh được với mô hình transformer — nhưng nó cho thấy hình dạng của bài toán: hai văn bản vào, nhãn 3 lớp ra, loss = cross-entropy trên `{entail, contradict, neutral}`.

## Cạm bẫy

- **Lối tắt chỉ dùng giả thuyết.** Mô hình có thể dự đoán nhãn chỉ từ giả thuyết với độ chính xác ~60% trên SNLI, vì "not", "nobody", "never" tương quan với contradiction. Đây là một baseline mạnh để phát hiện rò rỉ nhãn.
- **Heuristic trùng lặp từ vựng.** Heuristic dãy con ("mọi dãy con đều được kéo theo") qua được SNLI nhưng thất bại trên HANS/ANLI. Hãy dùng các bộ đánh giá đối kháng (adversarial example).
- **Suy giảm chất lượng với tài liệu dài.** Các mô hình NLI một câu tụt hơn 20 điểm F1 trên tiền đề dài bằng cả tài liệu. Dùng mô hình huấn luyện trên DocNLI cho ngữ cảnh dài.
- **Nhạy cảm với template zero-shot.** "This example is about {label}" so với "{label}" so với "The topic is {label}" có thể làm độ chính xác lệch hơn 10 điểm. Hãy tinh chỉnh template.
- **Lệch miền.** MNLI huấn luyện trên tiếng Anh phổ thông. Văn bản pháp lý, y khoa và khoa học cần mô hình NLI đặc thù miền (ví dụ SciNLI, MedNLI).

## Áp dụng

Bộ stack của năm 2026:

| Trường hợp dùng | Mô hình |
|---------|-------|
| NLI đa dụng | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Nhanh / thiết bị biên | `cross-encoder/nli-deberta-v3-base` |
| Phân loại zero-shot (nhẹ) | `facebook/bart-large-mnli` |
| NLI cấp tài liệu | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Đa ngôn ngữ | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Phát hiện hallucination trong RAG | Lớp NLI bên trong RAGAS / DeepEval |

Meta-pattern của năm 2026: NLI là băng dính của việc hiểu văn bản. Mỗi khi bạn cần "A có hỗ trợ B không?" hay "A có mâu thuẫn với B không?" — hãy với tay tới NLI trước khi nghĩ tới một lời gọi LLM khác.

## Đóng gói

Lưu thành `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Bài tập

1. **Dễ.** Chạy `facebook/bart-large-mnli` trên 20 bộ ba (tiền đề, giả thuyết, nhãn) tự tạo, phủ cả ba lớp. Đo độ chính xác. Thêm các bẫy "heuristic dãy con" đối kháng ("I did not eat the cake" vs "I ate the cake") và xem nó có vỡ không.
2. **Trung bình.** So sánh template zero-shot `"This text is about {label}"` với `"The topic is {label}"` và `"{label}"` trên 100 tiêu đề AG News. Báo cáo độ lệch chính xác.
3. **Khó.** Xây một bộ kiểm tra faithfulness cho RAG: tách câu trả lời thành các khẳng định nguyên tử + NLI cho từng khẳng định. Đánh giá trên 50 câu trả lời do RAG sinh kèm ngữ cảnh vàng. Đo tỉ lệ dương tính giả và âm tính giả so với nhãn do người gán.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| NLI | Suy luận ngôn ngữ tự nhiên | Phân loại 3 lớp quan hệ giữa tiền đề và giả thuyết. |
| RTE | Nhận dạng kéo theo văn bản | Tên cũ của NLI; cùng một bài toán. |
| Entailment | "t ngụ ý h" | Một người đọc điển hình sẽ kết luận h đúng khi có t. |
| Contradiction | "t loại trừ h" | Một người đọc điển hình sẽ kết luận h sai khi có t. |
| Neutral | "chưa quyết định" | Không suy ra được gì từ t sang h theo cả hai hướng. |
| Zero-shot classification | NLI làm bộ phân loại | Diễn đạt nhãn thành giả thuyết, chọn kéo theo cao nhất. |
| Faithfulness | Câu trả lời có được hỗ trợ không? | NLI trên (ngữ cảnh truy xuất, câu trả lời được sinh). |

## Đọc thêm

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — benchmark ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI làm bộ phân loại.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — cỗ máy NLI của năm 2026.

---

[← Bài trước: Đầu ra có cấu trúc & Giải mã có ràng buộc](../../20-structured-outputs-constrained-decoding/docs/vi.md) · [Bài sau: Mô hình embedding (embedding model) →](../../22-embedding-models-deep-dive/docs/vi.md)
