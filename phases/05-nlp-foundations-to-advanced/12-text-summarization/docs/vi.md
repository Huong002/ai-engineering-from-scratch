# Tóm tắt văn bản

> Hệ thống extractive (trích rút) cho bạn biết tài liệu đã nói gì. Hệ thống abstractive (trừu tượng) cho bạn biết tác giả muốn nói gì. Nhiệm vụ khác nhau, cạm bẫy khác nhau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 02 (BoW + TF-IDF), Giai đoạn 5, Bài 11 (Dịch máy)
**Time:** ~75 phút

## Vấn đề

Một bài báo 2.000 từ xuất hiện trong feed của bạn. Bạn cần 120 từ để nắm được nó. Bạn có thể chọn ba câu quan trọng nhất từ bài báo (extractive) hoặc viết lại nội dung bằng lời của bạn (abstractive). Cả hai đều được gọi là summarization (tóm tắt). Chúng là hai bài toán hoàn toàn khác nhau.

Tóm tắt extractive là một bài toán xếp hạng. Chấm điểm từng câu, trả về top-`k`. Đầu ra luôn đúng ngữ pháp vì nó được lấy nguyên văn. Rủi ro là bỏ sót nội dung nằm rải rác khắp bài.

Tóm tắt abstractive là một bài toán sinh văn bản. Một transformer tạo ra văn bản mới dựa trên đầu vào. Đầu ra trôi chảy và nén tốt nhưng có thể bịa ra những sự kiện không có trong nguồn. Rủi ro là sự bịa đặt đầy tự tin.

Bài học này xây dựng cả hai, kèm theo kiểu thất bại mà mỗi cách mắc phải.

## Khái niệm

![TextRank trích rút so với transformer trừu tượng](../assets/summarization.svg)

**Extractive.** Coi bài báo như một đồ thị với các nút là câu và các cạnh là độ tương đồng. Chạy PageRank (hoặc thứ gì đó tương tự) trên đồ thị để chấm điểm câu theo mức độ chúng kết nối với mọi thứ khác. Những câu điểm cao nhất là bản tóm tắt. Cách hiện thực kinh điển là **TextRank** (Mihalcea và Tarau, 2004).

**Abstractive.** Fine-tune một transformer encoder-decoder (BART, T5, Pegasus) trên các cặp tài liệu-tóm tắt. Khi suy luận, mô hình đọc tài liệu và sinh bản tóm tắt từng token một thông qua cross-attention. Riêng Pegasus dùng mục tiêu tiền huấn luyện gap-sentence, khiến nó rất giỏi tóm tắt mà không cần fine-tune nhiều.

Đánh giá bằng **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). ROUGE-1 và ROUGE-2 chấm điểm trùng lặp unigram và bigram. ROUGE-L chấm điểm chuỗi con chung dài nhất. Cao hơn là tốt hơn, nhưng ROUGE-L 40 là "tốt" và 50 là "xuất sắc". Mọi bài báo đều báo cáo cả ba. Dùng gói `rouge-score`.

```figure
summarize-collapse
```

## Tự tay làm

### Bước 1: TextRank (extractive)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Hai điều đáng gọi tên. Hàm tương đồng dùng độ trùng lặp từ đã chuẩn hoá log, đây là biến thể TextRank gốc. Cosine của vector TF-IDF cũng dùng được. Hệ số tắt dần 0.85 và số vòng lặp là mặc định của PageRank.

### Bước 2: abstractive với BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN được fine-tune trên kho CNN/DailyMail. Nó tạo ra bản tóm tắt kiểu tin tức ngay từ đầu. Với lĩnh vực khác (bài báo khoa học, hội thoại, pháp lý), dùng checkpoint Pegasus tương ứng hoặc fine-tune trên dữ liệu đích của bạn.

### Bước 3: đánh giá ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Luôn dùng stemming. Không có nó, "running" và "run" bị tính là hai từ khác nhau và ROUGE đếm thiếu.

### Ngoài ROUGE (đánh giá tóm tắt năm 2026)

ROUGE đã là thước đo tóm tắt thống trị suốt hai mươi năm và đến 2026 nó không còn đủ nếu đứng một mình. Một phân tích tổng hợp quy mô lớn các bài báo NLG cho thấy:

- **BERTScore** (độ tương đồng embedding theo ngữ cảnh) đã chiếm được chỗ đứng trong suốt 2023 và giờ được báo cáo song song với ROUGE trong hầu hết bài báo về tóm tắt.
- **BARTScore** coi đánh giá là sinh văn bản: chấm điểm bản tóm tắt theo xác suất mà một BART đã tiền huấn luyện gán cho nó khi biết nguồn.
- **MoverScore** (Earth Mover's Distance trên embedding theo ngữ cảnh) đạt vị trí đầu trong các benchmark tóm tắt 2025 vì nó nắm được trùng lặp ngữ nghĩa tốt hơn ROUGE.
- **FactCC** và **QA-based faithfulness** phổ biến trong giai đoạn 2021-2023, nay thường được thay bằng **G-Eval** (một chuỗi prompt GPT-4 chấm điểm coherence, consistency, fluency, relevance kèm lập luận chain-of-thought).
- **G-Eval** và các cách tiếp cận LLM-judge tương tự khớp với đánh giá của con người ~80% số lần khi rubric được thiết kế tốt.

Khuyến nghị cho production: báo cáo ROUGE-L để so sánh với hệ cũ, BERTScore cho trùng lặp ngữ nghĩa, G-Eval cho coherence và tính đúng sự thật. Hiệu chỉnh với 50-100 bản tóm tắt do con người gán nhãn.

### Bước 4: vấn đề tính đúng sự thật

Bản tóm tắt abstractive dễ bịa. Bản tóm tắt extractive có rủi ro bịa thấp hơn nhiều vì đầu ra được lấy nguyên văn từ nguồn, dù chúng vẫn có thể gây nhầm lẫn nếu các câu nguồn bị tách khỏi ngữ cảnh, đã lỗi thời, hoặc được trích dẫn sai thứ tự. Đây là lý do lớn nhất khiến các hệ thống production vẫn ưu tiên phương pháp extractive cho nội dung gần với việc tuân thủ quy định.

Các kiểu bịa cần gọi tên:

- **Đổi thực thể.** Nguồn nói "John Smith." Bản tóm tắt nói "John Brown."
- **Lệch số liệu.** Nguồn nói "25,000." Bản tóm tắt nói "25 million."
- **Đảo cực tính.** Nguồn nói "rejected the offer." Bản tóm tắt nói "accepted the offer."
- **Bịa sự kiện.** Nguồn không nhắc tới CEO. Bản tóm tắt nói CEO đã phê duyệt.

Các cách đánh giá hiệu quả:

- **FactCC.** Một classifier nhị phân được huấn luyện trên quan hệ entailment giữa câu nguồn và câu tóm tắt. Dự đoán có sự thật/không có sự thật.
- **Đánh giá dựa trên QA.** Hỏi một mô hình QA những câu hỏi mà câu trả lời nằm trong nguồn. Nếu bản tóm tắt ủng hộ câu trả lời khác, gắn cờ.
- **F1 ở mức thực thể.** So sánh các thực thể có tên trong nguồn với trong bản tóm tắt. Thực thể chỉ xuất hiện trong bản tóm tắt là đáng ngờ.

Với bất cứ thứ gì hướng tới người dùng mà tính đúng sự thật quan trọng (tin tức, y tế, pháp lý, tài chính), extractive là mặc định an toàn hơn. Abstractive cần một bước kiểm tra tính đúng sự thật trong vòng lặp.

## Áp dụng

Bộ công cụ của năm 2026:

| Trường hợp dùng | Khuyến nghị |
|---------|-------------|
| Tin tức, tóm tắt 3-5 câu, tiếng Anh | `facebook/bart-large-cnn` |
| Bài báo khoa học | `google/pegasus-pubmed` hoặc một T5 đã tinh chỉnh |
| Nhiều tài liệu, dạng dài | Bất kỳ LLM nào có ngữ cảnh 32k+, đưa vào prompt |
| Tóm tắt hội thoại | `philschmid/bart-large-cnn-samsum` |
| Extractive, rủi ro bịa thấp ngay từ cấu trúc | TextRank hoặc LSA / LexRank của `sumy` |

Các LLM có ngữ cảnh dài thường vượt qua mô hình chuyên dụng vào năm 2026 khi compute không phải là ràng buộc. Đánh đổi là chi phí và khả năng tái lập; mô hình chuyên dụng cho đầu ra ổn định hơn.

## Đóng gói

Lưu thành `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Bài tập

1. **Dễ.** Chạy TextRank trên 5 bài báo. So sánh top-3 câu với một bản tóm tắt tham chiếu. Đo ROUGE-L. Bạn sẽ thấy 30-45 ROUGE-L trên các bài kiểu CNN/DailyMail.
2. **Trung bình.** Hiện thực đánh giá tính đúng sự thật ở mức thực thể: trích các thực thể có tên từ nguồn và bản tóm tắt (spaCy), tính recall của thực thể nguồn trong bản tóm tắt và precision của thực thể trong bản tóm tắt so với nguồn. Precision cao và recall thấp nghĩa là an toàn nhưng súc tích; precision thấp nghĩa là thực thể bị bịa.
3. **Khó.** So sánh BART-large-CNN với một LLM (Claude hoặc GPT-4) trên 50 bài CNN/DailyMail. Báo cáo ROUGE-L, tính đúng sự thật (bằng F1 thực thể), và chi phí mỗi bản tóm tắt. Ghi lại nơi mỗi bên thắng.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Extractive | Chọn câu | Trả về nguyên văn các câu từ nguồn. Không bao giờ bịa. |
| Abstractive | Viết lại | Sinh văn bản mới dựa trên nguồn. Có thể bịa. |
| ROUGE | Thước đo tóm tắt | Trùng lặp N-gram / LCS giữa đầu ra hệ thống và bản tham chiếu. |
| TextRank | Extractive dựa trên đồ thị | PageRank trên đồ thị tương đồng giữa các câu. |
| Factuality | Có đúng không | Liệu các khẳng định trong bản tóm tắt có được nguồn ủng hộ không. |
| Hallucination | Nội dung bịa ra | Nội dung trong bản tóm tắt mà nguồn không ủng hộ. |

## Đọc thêm

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — bài báo kinh điển về extractive.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — bài báo về BART.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus và mục tiêu gap-sentence.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — bài báo ROUGE.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — bài báo về toàn cảnh tính đúng sự thật.

---

[← Bài trước: Dịch máy](../../11-machine-translation/docs/vi.md) · [Bài sau: Hệ thống hỏi đáp →](../../13-question-answering/docs/vi.md)
