# Truy hồi thông tin (information retrieval) và Tìm kiếm

> BM25 chính xác nhưng dễ vỡ. Dense giăng lưới rộng nhưng bỏ sót từ khoá. Hybrid là mặc định của năm 2026. Phần còn lại chỉ là tinh chỉnh.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 02 (BoW + TF-IDF), Giai đoạn 5, Bài 04 (GloVe, FastText, Subword)
**Time:** ~75 phút

## Vấn đề

Người dùng gõ "what happens if someone lies to get money" và mong tìm ra điều luật thực sự bao phủ hành vi đó: "Section 420 IPC." Tìm kiếm theo từ khoá bỏ sót hoàn toàn (không có từ vựng chung). Tìm kiếm ngữ nghĩa bỏ sót nếu embedding không được huấn luyện trên văn bản pháp luật. Tìm kiếm thực tế phải xử lý cả hai.

IR là pipeline nằm dưới mọi hệ thống RAG, mọi thanh tìm kiếm, mọi tra cứu mờ của site tài liệu. Kiến trúc năm 2026 vận hành được trong thực tế không phải một phương pháp duy nhất. Nó là một chuỗi các phương pháp bổ trợ nhau, mỗi cái bắt lấy thất bại của cái trước.

Bài này dựng từng mảnh và gọi tên thất bại mà mỗi mảnh bắt được.

## Khái niệm

![Truy hồi hybrid: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Bốn lớp. Chọn những lớp bạn cần.

1. **Sparse retrieval (BM25).** Nhanh, chính xác khi khớp đúng, tệ về ngữ nghĩa. Chạy trên inverted index (chỉ mục nghịch đảo). Dưới 10ms mỗi query (truy vấn) trên hàng triệu tài liệu. Lấy đúng các tham chiếu điều luật, mã sản phẩm, thông báo lỗi, thực thể có tên.
2. **Dense retrieval (truy hồi dense).** Mã hoá query và tài liệu thành vector. Tìm hàng xóm gần nhất. Bắt được cách diễn đạt lại và tương đồng ngữ nghĩa. Bỏ sót khớp từ khoá chính xác mà chỉ lệch một ký tự. 50-200ms mỗi query với FAISS hoặc một vector DB.
3. **Fusion.** Gộp các danh sách xếp hạng (ranking) từ sparse và dense. Reciprocal Rank Fusion (RRF) là mặc định dễ dùng vì nó bỏ qua điểm thô (vốn nằm ở các thang khác nhau) và chỉ dùng vị trí hạng. Weighted fusion là một lựa chọn khi bạn biết một tín hiệu trội hơn trong miền của mình.
4. **Cross-encoder rerank.** Lấy top-30 từ fusion. Chạy một cross-encoder (query + tài liệu cùng nhau, chấm điểm từng cặp). Giữ top-5. Cross-encoder chậm hơn bi-encoder trên mỗi cặp nhưng chính xác hơn nhiều. Bạn bù lại bằng cách chỉ chạy chúng trên top-30.

Truy hồi ba chiều (BM25 + dense + learned-sparse như SPLADE) vượt hai chiều trên các benchmark 2026 nhưng cần hạ tầng cho chỉ mục learned-sparse. Với hầu hết đội, hai chiều cộng cross-encoder rerank là điểm ngọt.

```figure
gx-hybrid-retrieval
```

## Tự tay làm

### Bước 1: BM25 từ đầu

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Hai tham số đáng biết. `k1=1.5` điều khiển độ bão hoà tần suất từ; cao hơn nghĩa là trọng số lớn hơn cho việc lặp từ. `b=0.75` điều khiển length normalization (chuẩn hoá độ dài văn bản); 0 bỏ qua độ dài tài liệu, 1 chuẩn hoá hoàn toàn. Các giá trị mặc định là khuyến nghị của Robertson từ bài báo gốc và hiếm khi cần chỉnh.

### Bước 2: truy hồi dense với bi-encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

Chuẩn hoá L2 embedding để tích vô hướng bằng cosine. `all-MiniLM-L6-v2` là 384 chiều, nhanh, và đủ mạnh cho hầu hết truy hồi tiếng Anh. Với công việc đa ngữ, dùng `paraphrase-multilingual-MiniLM-L12-v2`. Để có độ chính xác cao nhất, dùng `bge-large-en-v1.5` hoặc `e5-large-v2`.

### Bước 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Hằng số `k=60` đến từ bài báo RRF gốc. `k` cao hơn làm phẳng đóng góp của khác biệt hạng; `k` thấp hơn khiến các hạng đầu trội hơn. 60 là mặc định đã công bố và hiếm khi cần chỉnh.

### Bước 4: tìm kiếm hybrid + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Ba tầng ghép lại. BM25 tìm khớp từ vựng. Dense tìm khớp ngữ nghĩa. RRF gộp hai bảng xếp hạng mà không cần hiệu chỉnh điểm. Cross-encoder chấm lại top-30 bằng cách đưa cặp query-tài liệu cùng nhau, nhờ đó bắt được độ liên quan mịn mà bi-encoder bỏ sót. Giữ top-5.

### Bước 5: đánh giá

| Chỉ số | Ý nghĩa |
|--------|---------|
| Recall@k | Trong số các query mà tài liệu đúng tồn tại, tần suất nó nằm trong top-k là bao nhiêu? |
| MRR (Mean Reciprocal Rank) | Trung bình của 1/hạng của tài liệu liên quan đầu tiên. |
| nDCG@k | Tính đến các mức độ liên quan, không chỉ nhị phân liên quan/không. |

Với RAG cụ thể, **Recall@k** (độ phủ @k) của retriever là con số quan trọng nhất. Reader của bạn không thể trả lời nếu đoạn đúng không nằm trong tập đã truy hồi.

Mẹo gỡ lỗi: với các query thất bại, diff bảng xếp hạng sparse và dense. Nếu một bên tìm ra tài liệu đúng còn bên kia không, bạn có lệch từ vựng (sửa: thêm nửa còn thiếu) hoặc nhập nhằng ngữ nghĩa (sửa: embedding tốt hơn hoặc một reranker).

## Áp dụng

Stack của năm 2026:

| Quy mô | Stack |
|-------|-------|
| 1k-100k docs | BM25 in-memory + embedding `all-MiniLM-L6-v2` + RRF. Không cần DB riêng. |
| 100k-10M docs | FAISS hoặc pgvector cho dense + Elasticsearch / OpenSearch cho BM25. Chạy song song. |
| 10M+ docs | Qdrant / Weaviate / Vespa / Milvus có hỗ trợ hybrid. Cross-encoder rerank trên top-30. |
| Best-quality frontier | Ba chiều (BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

Dù bạn chọn gì, hãy dành ngân sách cho đánh giá. Benchmark recall (độ phủ) của truy hồi trước khi benchmark độ chính xác RAG đầu-cuối. Reader không thể sửa thứ retriever đã bỏ sót.

### Những bài học xương máu từ RAG production năm 2026

- **80% lỗi RAG bắt nguồn từ ingestion và chunking, không phải mô hình.** Các đội dành hàng tuần đổi LLM và tinh chỉnh prompt trong khi truy hồi âm thầm trả về ngữ cảnh sai ở mỗi query thứ ba. Sửa chunking trước.
- **Chiến lược chunking quan trọng hơn kích thước chunk.** Cắt theo kích thước cố định làm vỡ bảng, code và heading lồng nhau. Chunking theo câu là mặc định; chunking ngữ nghĩa hoặc dựa trên LLM trả công cho tài liệu kỹ thuật và hướng dẫn sản phẩm.
- **Mẫu parent-doc.** Truy hồi các chunk "con" nhỏ để có precision (độ chính xác). Khi nhiều con từ cùng một mục cha xuất hiện, thay bằng khối cha để giữ ngữ cảnh. Điều này nâng chất lượng câu trả lời một cách ổn định mà không cần huấn luyện lại.
- **k_rerank=3 thường là tối ưu.** Mỗi chunk thêm sau đó tăng chi phí token và độ trễ sinh mà không nâng chất lượng câu trả lời. Nếu k=8 vẫn tốt hơn k=3 với bạn, reranker đang hoạt động dưới mức.
- **HyDE / query expansion.** Sinh một câu trả lời giả định từ query, embed nó, rồi truy hồi. Bắc cầu khoảng cách diễn đạt giữa câu hỏi ngắn và tài liệu dài. Nâng precision miễn phí, không cần huấn luyện.
- **Ngân sách ngữ cảnh dưới 8K token.** Trúng liên tục ở ngưỡng đó nghĩa là ngưỡng reranker quá lỏng.
- **Đánh version mọi thứ.** Prompt, quy tắc chunking, mô hình embedding, reranker. Bất kỳ trôi lệch nào cũng âm thầm phá chất lượng câu trả lời. Các cổng CI về faithfulness, context precision và tỉ lệ câu hỏi không trả lời được sẽ chặn hồi quy trước khi người dùng thấy.
- **Truy hồi ba chiều (BM25 + dense + learned-sparse như SPLADE) vượt hai chiều** trên benchmark 2026, đặc biệt với query trộn danh từ riêng với ngữ nghĩa. Hãy ship khi hạ tầng hỗ trợ chỉ mục SPLADE.

Thiết kế truy hồi đúng giảm ảo giác 70-90% theo đo lường công nghiệp năm 2026. Hầu hết lợi ích hiệu năng RAG đến từ truy hồi tốt hơn, không phải fine-tune mô hình.

## Đóng gói

Lưu thành `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Bài tập

1. **Dễ.** Cài đặt `hybrid_search` ở trên trên một corpus 500 tài liệu. Thử 20 query. So sánh recall@5 giữa chỉ BM25, chỉ dense và hybrid.
2. **Trung bình.** Thêm tính toán MRR. Với mỗi query test có tài liệu đúng đã biết, tìm hạng của tài liệu đúng trong bảng xếp hạng BM25, dense và hybrid. Báo MRR cho từng cái.
3. **Khó.** Fine-tune một encoder dense trên miền của bạn bằng MultipleNegativesRankingLoss (Sentence Transformers). Dựng tập huấn luyện từ 500 cặp query-tài liệu. So sánh recall trước và sau fine-tune.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| BM25 | Tìm kiếm từ khoá | Okapi BM25. Chấm điểm tài liệu theo tần suất từ, IDF và độ dài. |
| Dense retrieval | Tìm kiếm vector | Mã hoá query + doc thành vector, tìm hàng xóm gần nhất. |
| Bi-encoder | Mô hình embedding | Mã hoá query và doc độc lập. Nhanh lúc truy vấn. |
| Cross-encoder | Mô hình reranker | Mã hoá query + doc cùng nhau. Chậm nhưng chính xác. |
| RRF | Gộp bảng xếp hạng | Gộp hai bảng xếp hạng bằng cách cộng `1/(k + rank)`. |
| Recall@k | Chỉ số truy hồi | Tỉ lệ query mà một tài liệu liên quan nằm trong top-k. |

## Đọc thêm

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — bài trình bày BM25 đầy đủ nhất.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, bi-encoder kinh điển.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — retriever learned-sparse thu hẹp khoảng cách với dense.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — bài báo RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — truy hồi late-interaction.

---

[← Bài trước: Hệ thống hỏi đáp](../../13-question-answering/docs/vi.md) · [Bài sau: Mô hình hoá chủ đề →](../../15-topic-modeling/docs/vi.md)
