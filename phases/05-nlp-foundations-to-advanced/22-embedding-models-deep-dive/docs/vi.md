# Mô hình embedding (embedding model) — Bản phân tích sâu 2026

> Word2Vec cho bạn một vector cho mỗi từ. Các mô hình embedding hiện đại cho bạn một vector cho mỗi đoạn văn, đa ngôn ngữ, với các góc nhìn thưa, dày và đa vector, kích thước vừa khít chỉ mục của bạn. Chọn sai là RAG của bạn truy hồi nhầm thứ.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 03 (Word2Vec), Giai đoạn 5, Bài 14 (Truy hồi thông tin)
**Time:** ~60 phút

## Vấn đề

Hệ thống RAG của bạn truy hồi sai đoạn văn trong 40% số lần. Thủ phạm hiếm khi là cơ sở dữ liệu vector hay prompt. Đó là mô hình embedding.

Chọn embedding năm 2026 nghĩa là chọn trên năm trục:

1. **Dày vs thưa vs đa vector.** Một vector cho mỗi đoạn, hoặc một vector cho mỗi token, hoặc một túi từ có trọng số thưa.
2. **Độ phủ ngôn ngữ.** Mô hình đơn ngữ tiếng Anh vẫn thắng trên tác vụ chỉ tiếng Anh. Mô hình đa ngôn ngữ thắng khi kho ngữ liệu trộn nhiều ngôn ngữ.
3. **Độ dài ngữ cảnh.** 512 token vs 8,192 vs 32,768 — và năng lực hiệu dụng thực tế thường chỉ 60-70% mức tối đa được quảng cáo.
4. **Ngân sách số chiều (dimension).** 3,072 số thực ở độ chính xác đầy đủ = 12 KB mỗi vector. Với 100M vector, lưu trữ tốn 1,300 USD/tháng. Cắt Matryoshka giảm 4×.
5. **Mở vs dịch vụ lưu trữ.** Trọng số mở nghĩa là bạn kiểm soát stack và dữ liệu. Dịch vụ lưu trữ nghĩa là bạn đổi quyền kiểm soát lấy bản mới nhất.

Bài này gọi tên các đánh đổi để bạn chọn dựa trên bằng chứng, không dựa trên thứ gì đang thịnh hành quý trước.

## Khái niệm

![Embedding dày, thưa và đa vector](../assets/embedding-modes.svg)

**Embedding dày.** Một vector cho mỗi đoạn văn (thường 384-3,072 chiều). Độ tương đồng cosine xếp hạng đoạn văn theo độ gần ngữ nghĩa. OpenAI `text-embedding-3-large`, chế độ dense của BGE-M3, Voyage-3. Lựa chọn mặc định.

**Embedding thưa.** Kiểu SPLADE. Một transformer dự đoán trọng số cho mỗi token trong từ vựng, rồi đặt phần lớn về 0. Kết quả là một vector thưa kích thước |vocab|. Nắm bắt khớp từ vựng (như BM25) nhưng với trọng số học được. Mạnh với truy vấn nặng từ khoá.

**Đa vector (tương tác muộn).** ColBERTv2, Jina-ColBERT. Một vector cho mỗi token. Chấm điểm bằng MaxSim: với mỗi query token, tìm token tài liệu giống nhất, rồi cộng điểm lại. Lưu và chấm tốn kém hơn, nhưng thắng với truy vấn dài và kho ngữ liệu chuyên ngành.

**BGE-M3: cả ba cùng lúc.** Một mô hình duy nhất xuất đồng thời biểu diễn dày, thưa và đa vector. Mỗi loại có thể được truy vấn độc lập; điểm số hợp nhất bằng tổng có trọng số. Mặc định của năm 2026 khi bạn muốn linh hoạt từ một checkpoint.

**Matryoshka Representation Learning.** Được huấn luyện để N chiều đầu của vector tự nó tạo thành một embedding nhỏ dùng được độc lập. Cắt một vector 1,536 chiều xuống 256 chiều và trả giá ~1% độ chính xác để tiết kiệm 6× lưu trữ. Được hỗ trợ bởi OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Bảng xếp hạng MTEB chỉ kể một phần câu chuyện

Massive Text Embedding Benchmark — 56 tác vụ thuộc 8 loại tác vụ khi ra mắt (2022), mở rộng lên hơn 100 tác vụ trong MTEB v2. Đầu năm 2026, Gemini Embedding 2 đứng đầu truy hồi (67.71 MTEB-R). Cohere embed-v4 dẫn đầu nhóm tổng quát (65.2 MTEB). BGE-M3 dẫn đầu nhóm trọng số mở đa ngôn ngữ (63.0). Bảng xếp hạng là cần thiết nhưng chưa đủ — luôn đánh giá trên lĩnh vực của bạn.

### Mẫu ba tầng

| Trường hợp dùng | Mẫu |
|----------|---------|
| Lượt đầu nhanh | Bi-encoder (bộ mã hoá hai nhánh) dày (BGE-M3, text-3-small) |
| Tăng recall | Thưa (SPLADE, BGE-M3 sparse) + hợp nhất RRF |
| Chính xác trên top-50 | Đa vector (ColBERTv2) hoặc cross-encoder (bộ mã hoá chéo) reranker (bộ xếp hạng lại) |

Hầu hết stack production dùng cả ba.

```figure
gx-matryoshka
```

## Tự tay làm

### Bước 1: đường cơ sở — embedding dày với Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` làm cho tích vô hướng bằng đúng độ tương đồng cosine. Luôn đặt nó.

### Bước 2: cắt Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Chuẩn hoá lại sau khi cắt. Nomic v1.5, OpenAI text-3 và Voyage-4 được huấn luyện để việc này không mất mát ở vài mức đầu tiên. Các mô hình không phải Matryoshka (Sentence-BERT gốc) suy giảm mạnh khi bị cắt.

### Bước 3: đa chức năng của BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Ba chỉ mục, một lần suy luận. Hợp nhất điểm:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Tinh chỉnh trọng số trên lĩnh vực của bạn.

### Bước 4: đánh giá MTEB trên một tác vụ tuỳ chỉnh

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Chạy các mô hình ứng viên trên một tập con *tiêu biểu*. Đừng tin thứ hạng bảng xếp hạng một cách đơn thuần — lĩnh vực của bạn mới là điều quan trọng.

### Bước 5: tự tay viết cosine từ đầu

Xem `code/main.py`. Embedding bằng Averaged Hashing Trick (chỉ dùng thư viện chuẩn). Không cạnh tranh được với embedding của transformer, nhưng cho thấy hình dạng: tokenize → vector → chuẩn hoá → tích vô hướng.

## Cạm bẫy

- **Dùng cùng một mô hình cho query và tài liệu.** Một số mô hình (Voyage, Jina-ColBERT) dùng mã hoá bất đối xứng — query và tài liệu đi qua các đường khác nhau. Luôn kiểm tra model card.
- **Thiếu tiền tố.** Các mô hình `bge-*` cần `"Represent this sentence for searching relevant passages: "` đặt trước query. Quên là mất 3-5 điểm recall.
- **Cắt Matryoshka quá tay.** 1,536 → 256 thường an toàn. 1,536 → 64 thì không. Hãy kiểm chứng trên tập đánh giá của bạn.
- **Cắt cụt ngữ cảnh.** Hầu hết mô hình âm thầm cắt ngắn đầu vào vượt quá độ dài tối đa. Tài liệu dài cần được chia khối (xem bài 23).
- **Bỏ qua đuôi độ trễ.** Điểm MTEB che giấu độ trễ p99. Một mô hình 600M có thể hơn mô hình 335M 2 điểm nhưng tốn gấp 3 lần cho mỗi truy vấn.

## Áp dụng

Stack của năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Chỉ tiếng Anh, nhanh, API | `text-embedding-3-large` hoặc `voyage-3-large` |
| Trọng số mở, tiếng Anh | `BAAI/bge-large-en-v1.5` |
| Trọng số mở, đa ngôn ngữ | `BAAI/bge-m3` hoặc `Qwen3-Embedding-8B` |
| Ngữ cảnh dài (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| Triển khai chỉ CPU | Nomic Embed v2 (137M tham số, MoE) |
| Hạn chế lưu trữ | Cắt Matryoshka + lượng tử hoá int8 |
| Truy vấn nặng từ khoá | Thêm SPLADE thưa, hợp nhất RRF với dày |

Mẫu của năm 2026: bắt đầu với BGE-M3 hoặc text-3-large, đánh giá trên lĩnh vực của bạn bằng MTEB, đổi mô hình nếu một mô hình chuyên ngành thắng hơn 3 điểm.

## Đóng gói

Lưu thành `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Bài tập

1. **Dễ.** Mã hoá 100 câu bằng `bge-small-en-v1.5` ở số chiều đầy đủ (384), rồi ở Matryoshka 128. Đo mức giảm MRR trên 10 truy vấn.
2. **Trung bình.** So sánh BGE-M3 dày, thưa và colbert trên 500 đoạn văn từ lĩnh vực của bạn. Cách nào thắng về recall@10? Hợp nhất RRF có thắng được chế độ đơn tốt nhất không?
3. **Khó.** Chạy MTEB trên ba mô hình ứng viên với hai tác vụ hàng đầu của lĩnh vực bạn. Báo cáo điểm MTEB, độ trễ p99 trên một batch 100 truy vấn, và USD/1M truy vấn. Chọn mô hình tối ưu Pareto.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Dense embedding | Cái vector | Một vector kích thước cố định cho mỗi văn bản. Độ tương đồng cosine để xếp hạng. |
| Sparse embedding | BM25 học được | Một trọng số cho mỗi token từ vựng; phần lớn bằng 0; huấn luyện đầu-cuối. |
| Multi-vector | Kiểu ColBERT | Một vector cho mỗi token; chấm điểm MaxSim; chỉ mục lớn hơn, recall tốt hơn. |
| Matryoshka | Mẹo búp bê Nga | N chiều đầu tự nó đã là một embedding nhỏ hợp lệ. |
| MTEB | Cái bảng xếp hạng | Massive Text Embedding Benchmark — 56 tác vụ khi ra mắt, hơn 100 trong v2. |
| BEIR | Bảng xếp hạng truy hồi | 18 tác vụ truy hồi zero-shot; thường được trích dẫn cho độ bền vững liên lĩnh vực. |
| Asymmetric encoding | Query ≠ đường của tài liệu | Mô hình dùng các phép chiếu khác nhau cho query và tài liệu. |

## Đọc thêm

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bài báo bi-encoder.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — bài báo của bảng xếp hạng.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — mô hình hợp nhất ba chế độ.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — mục tiêu huấn luyện theo thang số chiều.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — tương tác muộn trong production.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — bảng xếp hạng trực tiếp.

---

[← Bài trước: Natural Language Inference (suy luận ngôn ngữ tự nhiên)](../../21-nli-textual-entailment/docs/vi.md) · [Bài sau: Chiến lược chia khối (chunking) cho RAG →](../../23-chunking-strategies-rag/docs/vi.md)
