# Chiến lược chia khối (chunking) cho RAG

> Cấu hình chia khối ảnh hưởng tới chất lượng truy hồi ngang với việc chọn mô hình embedding (Vectara NAACL 2025). Chia khối sai thì xếp hạng lại (reranking) bao nhiêu cũng không cứu được.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 14 (Truy hồi thông tin), Giai đoạn 5, Bài 22 (Mô hình embedding)
**Time:** ~60 phút

## Vấn đề

Bạn đưa một hợp đồng 50 trang vào hệ thống RAG (retrieval-augmented generation — sinh văn bản tăng cường truy hồi). Người dùng hỏi: "Điều khoản chấm dứt là gì?" Bộ truy hồi trả về trang bìa. Tại sao? Vì mô hình được huấn luyện trên các khối 512 token, còn điều khoản chấm dứt nằm tận trang 20, bị cắt ngang qua một chỗ ngắt trang, và không có từ khoá cục bộ nào gắn nó với truy vấn.

Cách khắc phục không phải là "mua một mô hình embedding tốt hơn". Cách khắc phục là chunking. Khối lớn cỡ nào? Chồng lấn giữa khối (chunk overlap) ra sao? Cắt ở đâu? Có kèm ngữ cảnh xung quanh không?

Các benchmark tháng 2/2026 cho kết quả đáng ngạc nhiên:

- Nghiên cứu 2026 của Vectara: chunking đệ quy 512 token thắng chunking ngữ nghĩa, 69% → 54% độ chính xác.
- SPLADE + Mistral-8B trên Natural Questions: chồng lấn không mang lại lợi ích đo được nào.
- Vách ngữ cảnh (context cliff): chất lượng trả lời giảm mạnh quanh 2.500 token ngữ cảnh.

Câu trả lời "hiển nhiên" (chunking ngữ nghĩa, chồng lấn 20%, 1000 token) thường sai. Bài này xây trực giác cho sáu chiến lược và chỉ cho bạn biết khi nào nên dùng cái nào.

## Khái niệm

![Sáu chiến lược chia khối trên một đoạn văn](../assets/chunking.svg)

**Chunking cố định.** Cắt sau mỗi N ký tự hoặc token. Baseline đơn giản nhất. Cắt ngang câu. Nén tốt, mạch lạc kém.

**Đệ quy.** `RecursiveCharacterTextSplitter` của LangChain. Thử cắt theo `\n\n` trước, rồi `\n`, rồi `.`, rồi dấu cách. Dự phòng sạch sẽ. Mặc định của năm 2026.

**Ngữ nghĩa.** Embed từng câu. Tính độ tương đồng cosine giữa các câu liền kề. Cắt ở chỗ độ tương đồng tụt dưới một ngưỡng. Giữ được mạch lạc chủ đề. Chậm hơn; đôi khi tạo ra những mảnh 40 token li ti làm hại truy hồi.

**Theo câu.** Cắt theo ranh giới câu. Mỗi câu một khối, hoặc một cửa sổ N câu. Ngang ngửa chunking ngữ nghĩa tới ~5k token với chi phí chỉ bằng một phần nhỏ.

**Khối cha – khối con (parent-document).** Lưu các khối con nhỏ để truy hồi *và* khối cha lớn hơn để làm ngữ cảnh. Truy hồi bằng khối con; trả về khối cha. Suy giảm mềm: khối con tệ vẫn trả về khối cha tạm ổn.

**Chunking muộn (late chunking, 2024).** Embed toàn bộ tài liệu ở mức token trước, rồi gộp các embedding token thành embedding khối. Giữ được ngữ cảnh xuyên khối. Dùng được với các embedder ngữ cảnh dài (BGE-M3, Jina v3). Tốn tính toán hơn.

**Truy hồi có ngữ cảnh (contextual retrieval, Anthropic, 2024).** Gắn trước mỗi khối một bản tóm tắt do LLM sinh ra về vị trí của khối trong tài liệu ("Khối này là mục 3.2 của phần điều khoản chấm dứt..."). Cải thiện truy hồi 35-50% trong benchmark của chính Anthropic. Tốn kém khi lập chỉ mục.

### Nguyên tắc thắng mọi giá trị mặc định

Chọn kích thước khối khớp với loại truy vấn:

| Loại truy vấn | Kích thước khối |
|------------|-----------|
| Truy vấn dữ kiện ("tên CEO là gì?") | 256-512 token |
| Phân tích / nhiều bước | 512-1024 token |
| Hiểu toàn bộ một mục | 1024-2048 token |

Benchmark 2026 của NVIDIA. Khối phải đủ lớn để chứa câu trả lời cộng ngữ cảnh cục bộ, và đủ nhỏ để top-K mà bộ truy hồi trả về tập trung vào câu trả lời thay vì nhiễu ngữ cảnh.

```figure
n5-chunk-cuts
```

## Tự tay làm

### Bước 1: chunking cố định và đệ quy

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Bước 2: chunking ngữ nghĩa

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Tinh chỉnh `threshold` theo miền dữ liệu của bạn. Quá cao → vụn. Quá thấp → một khối khổng lồ.

### Bước 3: khối cha – khối con

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Điểm mấu chốt: khử trùng lặp khối cha. Nhiều khối con có thể trỏ về cùng một khối cha; trả về hết sẽ lãng phí ngữ cảnh.

### Bước 4: truy hồi có ngữ cảnh (mẫu của Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Lập chỉ mục các khối đã thêm ngữ cảnh. Lúc truy vấn, truy hồi hưởng lợi từ tín hiệu ngữ cảnh xung quanh thêm vào.

### Bước 5: đánh giá

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Luôn đo benchmark. Chiến lược "tốt nhất" cho kho dữ liệu của bạn có thể chẳng giống bài blog nào.

## Cạm bẫy

- **Chỉ đánh giá chunking trên truy vấn dữ kiện.** Truy vấn nhiều bước cho thấy người thắng rất khác. Dùng bộ đánh giá phân tầng theo loại truy vấn.
- **Chunking ngữ nghĩa không có kích thước tối thiểu.** Tạo ra các mảnh 40 token làm hại truy hồi. Luôn áp `min_tokens`.
- **Chồng lấn theo kiểu mê tín.** Các nghiên cứu 2026 thấy chồng lấn thường chẳng mang lại lợi ích gì mà còn gấp đôi chi phí lập chỉ mục. Hãy đo, đừng đoán.
- **Không áp min/max.** Khối 5 token hay 5000 token đều phá truy hồi. Hãy chặn hai đầu.
- **Chunking xuyên tài liệu.** Không bao giờ để một khối vắt qua hai tài liệu. Luôn chia khối theo từng tài liệu, rồi gộp lại.

## Áp dụng

Bộ công cụ của 2026:

| Tình huống | Chiến lược |
|-----------|----------|
| Lần dựng đầu tiên, chưa rõ kho dữ liệu | Đệ quy, 512 token, không chồng lấn |
| Hỏi đáp dữ kiện | Đệ quy, 256-512 token |
| Phân tích / nhiều bước | Đệ quy, 512-1024 token + khối cha – khối con |
| Nhiều tham chiếu chéo (hợp đồng, bài báo) | Chunking muộn hoặc truy hồi có ngữ cảnh |
| Kho hội thoại / đối thoại | Khối theo lượt nói + metadata người nói |
| Phát ngôn ngắn (tweet, đánh giá) | Một tài liệu = một khối |

Bắt đầu với đệ quy 512. Đo recall@k (độ phủ trong top-k) — cụ thể là recall@5 — trên bộ đánh giá 50 truy vấn. Rồi tinh chỉnh từ đó.

## Đóng gói

Lưu thành `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Bài tập

1. **Dễ.** Chia khối một tài liệu 20 trang bằng fixed(512, 0), recursive(512, 0) và recursive(512, 100). So sánh số khối và chất lượng ranh giới.
2. **Trung bình.** Dựng bộ đánh giá 30 truy vấn trên 5 tài liệu. Đo recall@5 cho đệ quy, ngữ nghĩa và khối cha – khối con. Cái nào thắng? Có khớp với các bài blog không?
3. **Khó.** Cài đặt truy hồi có ngữ cảnh. Đo mức cải thiện MRR so với baseline đệ quy. Báo cáo chi phí lập chỉ mục (số lần gọi LLM) so với mức tăng độ chính xác.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Chunk | Một mẩu tài liệu | Đơn vị dưới-tài-liệu được embed, lập chỉ mục và truy hồi. |
| Overlap | Biên an toàn | N token dùng chung giữa các khối liền kề; thường vô dụng trong benchmark 2026. |
| Semantic chunking | Chia khối thông minh | Cắt ở chỗ độ tương đồng embedding giữa các câu liền kề tụt xuống. |
| Parent-document | Truy hồi hai tầng | Truy hồi khối con nhỏ, trả về khối cha lớn hơn. |
| Late chunking | Chia khối sau khi embed | Embed cả tài liệu ở mức token, gộp thành vector khối. |
| Contextual retrieval | Chiêu của Anthropic | Bản tóm tắt do LLM sinh, gắn trước mỗi khối trước khi lập chỉ mục. |
| Context cliff | Bức tường 2500 token | Mức giảm chất lượng quan sát quanh 2.5k token ngữ cảnh trong RAG (1/2026). |

## Đọc thêm

- [Yepes et al. / LangChain — Tài liệu Recursive Character Splitting](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — mặc định trong môi trường production.
- [Vectara (2024, NAACL 2025). Phân tích cấu hình chunking](https://arxiv.org/abs/2410.13070) — chunking quan trọng ngang với việc chọn embedding.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — bài báo về chunking muộn.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — cải thiện truy hồi 35-50% nhờ tiền tố ngữ cảnh do LLM sinh.
- [NVIDIA 2026 chunk-size benchmark — bản tóm tắt của Premai](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — kích thước khối theo loại truy vấn.

---

[← Bài trước: Mô hình embedding (embedding model)](../../22-embedding-models-deep-dive/docs/vi.md) · [Bài sau: Phân giải đồng tham chiếu (coreference resolution) →](../../24-coreference-resolution/docs/vi.md)
