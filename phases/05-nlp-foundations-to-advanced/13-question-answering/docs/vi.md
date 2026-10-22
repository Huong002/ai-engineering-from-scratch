# Hệ thống hỏi đáp

> Ba hệ thống đã định hình question answering (hỏi đáp) hiện đại. Trích xuất tìm ra các span. Truy xuất tăng cường neo chúng vào tài liệu. Sinh tạo ra câu trả lời. Mọi trợ lý AI hiện đại đều là sự pha trộn của cả ba.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 11 (Dịch máy), Giai đoạn 5, Bài 10 (Cơ chế attention)
**Time:** ~75 phút

## Vấn đề

Một người dùng gõ "Khi nào chiếc iPhone đầu tiên ra mắt?" và mong đợi "29 tháng 6, 2007." Không phải "Lịch sử của Apple dài và nhiều biến động." Cũng không phải "2007" nằm trơ trọi mà không có câu nào. Mà là một câu trả lời trực tiếp, có căn cứ, chính xác.

Ba kiến trúc đã thống trị QA trong thập kỷ qua.

- **QA trích xuất (extractive QA).** Cho một câu hỏi và một context passage (đoạn ngữ cảnh) được biết là có chứa câu trả lời, hãy tìm chỉ số bắt đầu và kết thúc của đoạn trả lời — span extraction (trích đoạn trả lời) — bên trong đoạn văn đó. SQuAD là benchmark chuẩn cho reading comprehension (đọc hiểu).
- **QA miền mở (open-domain QA).** Đoạn văn không được đưa sẵn. Phải truy xuất đoạn văn liên quan trước, rồi trích xuất hoặc sinh câu trả lời. Đây là nền móng của mọi pipeline RAG ngày nay.
- **QA sinh / sách đóng (generative / closed-book QA).** Một mô hình ngôn ngữ lớn trả lời từ bộ nhớ tham số của nó. Không truy xuất. Nhanh nhất khi suy luận, kém tin cậy nhất về sự kiện.

Xu hướng năm 2026 là kết hợp: truy xuất vài đoạn văn tốt nhất, rồi prompt một mô hình sinh trả lời dựa trên chính các đoạn văn đó. Đó là RAG, và bài 14 đi sâu vào nửa phần truy xuất. Bài này xây nửa phần QA.

## Khái niệm

![Kiến trúc hỏi đáp: trích xuất, tăng cường truy hồi, sinh văn bản](../assets/qa.svg)

**Trích xuất.** Mã hoá câu hỏi và đoạn văn cùng nhau bằng một transformer (họ BERT). Huấn luyện hai head xuất ra start/end logits (logit đầu/cuối) để dự đoán chỉ số token bắt đầu và kết thúc của câu trả lời. Loss là cross-entropy trên các vị trí hợp lệ. Đầu ra là một span từ đoạn văn. Không bao giờ bịa đặt (do cách xây dựng), cũng không bao giờ xử lý được câu hỏi mà đoạn văn không thể trả lời (do cách xây dựng).

**Truy xuất tăng cường (RAG).** Hai tầng. Đầu tiên, một retriever tìm top-`k` đoạn văn từ một kho. Thứ hai, một reader (trích xuất hoặc sinh) tạo câu trả lời dựa trên các đoạn văn đó. Việc tách retriever–reader cho phép huấn luyện và đánh giá từng phần độc lập. RAG hiện đại thường thêm một reranker ở giữa hai phần.

**Sinh.** Một LLM chỉ có decoder (GPT, Claude, Llama) trả lời từ trọng số đã học. Không có bước truy xuất. Rất tốt với kiến thức phổ thông, thảm hoạ với sự kiện hiếm hoặc mới. Tỷ lệ hallucination tỷ lệ nghịch với tần suất của sự kiện trong dữ liệu tiền huấn luyện.

```figure
qa-span
```

## Tự tay làm

### Bước 1: QA trích xuất với một mô hình đã huấn luyện sẵn

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` được huấn luyện trên SQuAD 2.0, bộ dữ liệu có chứa cả những unanswerable question (câu hỏi không trả lời được). Mặc định, pipeline `question-answering` trả về span có điểm cao nhất kể cả khi điểm null của mô hình thắng — nó *không* tự động trả về câu trả lời rỗng. Để có hành vi "không có câu trả lời" rõ ràng, truyền `handle_impossible_answer=True` vào lời gọi pipeline: khi đó pipeline chỉ trả về câu trả lời rỗng nếu điểm null vượt mọi điểm span. Dù thế nào cũng luôn kiểm tra trường `score`.

### Bước 2: một pipeline truy xuất tăng cường (phác thảo)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Pipeline hai tầng. Dense retriever (Sentence-BERT) tìm các đoạn văn liên quan bằng độ tương đồng ngữ nghĩa. Extractive reader (RoBERTa-SQuAD) rút span câu trả lời từ các đoạn văn đứng đầu đã được ghép lại. Chạy tốt trên kho nhỏ. Với kho hàng triệu tài liệu, hãy dùng FAISS hoặc một vector database.

### Bước 3: sinh với RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Mẫu prompt rất quan trọng. Nói rõ với mô hình rằng phải bám vào ngữ cảnh và trả về "I don't know." khi ngữ cảnh không đủ giúp giảm tỷ lệ hallucination 40-60% so với cách prompt ngây thơ. Những mẫu cầu kỳ hơn thêm trích dẫn, điểm tin cậy và trích xuất có cấu trúc.

### Bước 4: đánh giá phản ánh đúng thực tế

SQuAD dùng **Exact Match (EM)** (khớp chính xác) và **F1 ở mức token**. EM là khớp nghiêm ngặt sau khi chuẩn hoá (về chữ thường, bỏ dấu câu, bỏ mạo từ) — hoặc dự đoán khớp chính xác, hoặc được 0. F1 được tính trên phần token trùng nhau giữa dự đoán và đáp án tham chiếu, và cho điểm một phần. Cả hai đều đánh giá thấp những cách diễn đạt lại: "June 29, 2007" so với "June 29th, 2007" thường được 0 EM (phần số thứ tự phá vỡ bước chuẩn hoá) nhưng vẫn được F1 đáng kể nhờ các token trùng nhau.

Với QA trong production:

- **Độ chính xác câu trả lời** (do LLM chấm hoặc người chấm, vì các metric không bắt được sự tương đương ngữ nghĩa).
- **Độ chính xác trích dẫn.** Đoạn văn được trích dẫn có thực sự chống lưng cho câu trả lời không? Rất dễ kiểm tra tự động bằng so khớp chuỗi giữa trích dẫn được sinh ra và các đoạn văn truy xuất được.
- **Hiệu chuẩn từ chối.** Khi câu trả lời không nằm trong các đoạn văn truy xuất được, hệ thống có nói đúng "I don't know" không? Hãy đo tỷ lệ tự tin sai.
- **Recall truy xuất.** Trước khi đánh giá reader, hãy đo xem retriever có đưa đúng đoạn văn vào top-`k` không. Reader không thể sửa được một đoạn văn bị thiếu.

### RAGAS: khung đánh giá production năm 2026

`RAGAS` được xây dựng riêng cho hệ RAG và là lựa chọn mặc định khi triển khai trong năm 2026. Nó chấm bốn chiều mà không cần đáp án vàng:

- **Faithfulness (trung thực).** Mỗi khẳng định trong câu trả lời có đến từ ngữ cảnh truy xuất được không? Đo bằng entailment dựa trên NLI. Đây là metric hallucination chính của bạn.
- **Answer relevance (liên quan câu hỏi).** Câu trả lời có trả lời đúng câu hỏi không? Đo bằng cách sinh các câu hỏi giả định từ câu trả lời rồi so với câu hỏi thật.
- **Context precision (độ chính xác ngữ cảnh).** Trong các chunk truy xuất được, bao nhiêu phần thực sự liên quan? Precision thấp = prompt có nhiễu.
- **Context recall (độ phủ ngữ cảnh).** Tập truy xuất được có chứa đủ thông tin cần thiết không? Recall thấp = reader không thể thành công.

Chấm điểm không cần đáp án tham chiếu cho phép bạn đánh giá trên lưu lượng production thật mà không cần bộ đáp án vàng được tuyển chọn. Thêm LLM-as-judge lên trên cho các câu hỏi mở, nơi metric khớp chính xác là vô dụng.

`pip install ragas`. Cắm retriever + reader của bạn vào. Nhận bốn số vô hướng mỗi truy vấn. Đặt cảnh báo khi có hồi quy.

## Áp dụng

Bộ công cụ năm 2026.

| Trường hợp dùng | Đề xuất |
|---------|-------------|
| Cho sẵn đoạn văn, tìm span câu trả lời | `deepset/roberta-base-squad2` |
| Trên một kho cố định, không chấp nhận sách đóng | RAG: dense retriever + LLM reader |
| Thời gian thực trên một kho tài liệu | RAG với retriever lai (BM25 + dense) + reranker (bài 14) |
| QA hội thoại (câu hỏi tiếp nối) | LLM có lịch sử hội thoại + RAG ở mỗi lượt |
| Miền đặc thù, cần độ chính xác cao, có quy định | Trích xuất trên một kho có thẩm quyền; không bao giờ chỉ dùng sinh |

QA trích xuất không còn thời thượng trong năm 2026 vì RAG với LLM xử lý được nhiều trường hợp hơn. Nó vẫn được triển khai ở những nơi bắt buộc phải trích dẫn nguyên văn: nghiên cứu pháp lý, tuân thủ quy định, công cụ kiểm toán.

## Đóng gói

Lưu thành `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Bài tập

1. **Dễ.** Dựng pipeline QA trích xuất SQuAD ở trên trên 10 đoạn Wikipedia. Tự tay soạn 10 câu hỏi. Đo xem câu trả lời đúng bao nhiêu lần. Bạn sẽ thấy 7-9 câu đúng nếu các đoạn văn và câu hỏi đều sạch.
2. **Trung bình.** Thêm một bộ phân loại từ chối. Khi điểm truy xuất cao nhất dưới một ngưỡng (ví dụ cosine 0.3), trả về "I don't know" thay vì gọi reader. Tinh chỉnh ngưỡng trên một tập held-out.
3. **Khó.** Xây một pipeline RAG trên kho 10.000 tài liệu do bạn chọn. Cài đặt truy xuất lai (BM25 + dense) với hợp nhất RRF (xem bài 14). Đo độ chính xác câu trả lời khi có và khi không có bước lai. Ghi lại loại câu hỏi nào hưởng lợi nhiều nhất.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Extractive QA | Tìm span câu trả lời | Dự đoán chỉ số bắt đầu và kết thúc của câu trả lời trong một đoạn văn cho sẵn. |
| Open-domain QA | QA trên một kho tài liệu | Không có đoạn văn cho sẵn; phải truy xuất rồi mới trả lời. |
| RAG | Truy xuất rồi sinh | Retrieval-augmented generation. Pipeline retriever + reader. |
| SQuAD | Benchmark chuẩn | Stanford Question Answering Dataset. Metric EM + F1. |
| Hallucination | Câu trả lời bịa ra | Đầu ra của reader không được ngữ cảnh truy xuất được chống lưng. |
| Refusal calibration | Biết khi nào nên im lặng | Hệ thống nói đúng "I don't know" khi không thể trả lời. |

## Đọc thêm

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — bài báo benchmark.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, dense retriever chuẩn cho QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — bài báo đặt tên cho RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — khảo sát RAG toàn diện.

---

[← Bài trước: Tóm tắt văn bản](../../12-text-summarization/docs/vi.md) · [Bài sau: Truy hồi thông tin (information retrieval) và Tìm kiếm →](../../14-information-retrieval-search/docs/vi.md)
