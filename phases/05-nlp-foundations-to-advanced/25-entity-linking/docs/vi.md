# Liên kết thực thể & Khử nhập nhằng

> NER đã tìm ra "Paris." Entity linking (liên kết thực thể) quyết định: Paris, Pháp? Paris Hilton? Paris, Texas? Paris (hoàng tử thành Troy)? Không có bước liên kết, knowledge graph của bạn vẫn mơ hồ.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 06 (NER), Giai đoạn 5, Bài 24 (Phân giải đồng tham chiếu)
**Time:** ~60 phút

## Vấn đề

Một câu như sau: "Jordan beat the press." NER của bạn gắn "Jordan" là PERSON. Tốt. Nhưng *Jordan* nào?

- Michael Jordan (bóng rổ)?
- Michael B. Jordan (diễn viên)?
- Michael I. Jordan (giáo sư ML ở Berkeley — đúng, sự nhầm lẫn này có thật trong các bài báo ML)?
- Jordan (quốc gia)?
- Jordan (tên riêng gốc Hebrew)?

Entity linking (EL) phân giải mỗi mention về một mục duy nhất trong knowledge base (cơ sở tri thức): Wikidata, Wikipedia, DBpedia, hoặc KB miền của bạn. Hai bài toán con:

1. **Candidate generation (sinh ứng viên).** Với "Jordan," những mục KB nào là khả dĩ?
2. **Disambiguation (khử nhập nhằng).** Với ngữ cảnh này, ứng viên nào là đúng?

Cả hai bước đều học được. Cả hai đều có benchmark. Pipeline kết hợp đã ổn định suốt một thập kỷ — thứ thay đổi là chất lượng của bộ khử nhập nhằng.

## Khái niệm

![Pipeline liên kết thực thể: mention → ứng viên → thực thể đã khử nhập nhằng](../assets/entity-linking.svg)

**Candidate generation.** Với dạng bề mặt của mention ("Jordan"), tra ứng viên trong một alias index (chỉ mục bí danh). Từ điển bí danh Wikipedia phủ hầu hết thực thể có tên: "JFK" → John F. Kennedy, Jacqueline Kennedy, sân bay JFK, JFK (phim). Chỉ mục điển hình trả về 10-30 ứng viên cho mỗi mention.

**Disambiguation: ba cách tiếp cận.**

1. **Prior probability (xác suất tiên nghiệm) + ngữ cảnh (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)` — tức xác suất tiên nghiệm nhân với độ tương đồng ngữ cảnh (contextual similarity). Hiệu quả, nhanh, không cần training.
2. **Dựa trên embedding (ESS / REL / Blink).** Mã hoá mention + ngữ cảnh. Mã hoá mô tả của từng ứng viên. Lấy cosine lớn nhất. Mặc định của giai đoạn 2020-2024.
3. **Sinh (GENRE, 2021; dựa trên LLM, 2023+).** Giải mã tên chuẩn tắc của thực thể theo từng token. Bị ràng buộc vào một trie các tên thực thể hợp lệ nên đầu ra chắc chắn là một KB id hợp lệ.

**End-to-end vs pipeline.** Các mô hình hiện đại (ELQ, BLINK, ExtEnD, GENRE) chạy NER + candidate generation + khử nhập nhằng trong một lượt. Hệ pipeline vẫn chiếm ưu thế trong production vì bạn có thể thay từng thành phần.

### Hai phép đo

- **Mention recall (candidate gen).** Tỉ lệ các mention vàng mà mục KB đúng xuất hiện trong danh sách ứng viên. Đây là sàn của toàn pipeline.
- **Độ chính xác khử nhập nhằng / F1.** Khi đã có ứng viên đúng, top-1 đúng bao nhiêu lần.

Luôn báo cáo cả hai. Một hệ có 99% khử nhập nhằng trên 80% mention recall thực chất là pipeline 80%.

```figure
gx-entity-linking
```

## Tự tay làm

### Bước 1: dựng alias index từ redirect của Wikipedia

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Dữ liệu alias Wikipedia: ~18M cặp (alias, entity). Tải từ dump Wikidata. Lưu dưới dạng inverted index.

### Bước 2: khử nhập nhằng dựa trên ngữ cảnh

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Độ trùng lặp Jaccard chỉ là đồ chơi. Hãy thay bằng cosine similarity trên embedding (xem bước 2 trong `code/main.py` cho phiên bản transformer).

### Bước 3: dựa trên embedding (kiểu BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Ở thời điểm index, embed mỗi thực thể KB một lần. Ở thời điểm query, embed mention + ngữ cảnh một lần, dot-product với pool ứng viên, chọn max.

### Bước 4: liên kết thực thể sinh (khái niệm)

GENRE giải mã tiêu đề Wikipedia của thực thể theo từng ký tự. Constrained decoding (xem bài 20) bảo đảm chỉ những tiêu đề hợp lệ mới có thể được xuất ra. Tích hợp chặt với một trie có KB chống lưng. Hậu duệ hiện đại là REL-GEN và EL bằng prompt LLM với structured output.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Kết hợp với một whitelist (`choice` của Outlines), đây là pipeline EL đơn giản nhất để ship trong năm 2026.

### Bước 5: đánh giá trên AIDA-CoNLL

AIDA-CoNLL là benchmark EL chuẩn: 1,393 bài Reuters, 34k mention, thực thể Wikipedia. Báo cáo độ chính xác trong KB (`P@1`) và tỉ lệ phát hiện NIL (không có trong KB) ở ngoài KB.

## Cạm bẫy

- **Xử lý NIL.** Một số mention không có trong KB (thực thể mới nổi, người ít tiếng tăm). Hệ phải dự đoán NIL thay vì đoán bừa một thực thể sai. Đo riêng.
- **Lỗi biên mention.** NER thượng nguồn bỏ sót span một phần ("Bank of America" chỉ được gắn là "Bank"). Recall của EL tụt.
- **Thiên lệ phổ biến.** Hệ đã huấn luyện dự đoán quá mức các thực thể phổ biến. Một mention "Michael I. Jordan" trên bài báo ML thường được link tới Jordan bóng rổ.
- **EL đa ngữ.** Ánh xạ mention trong văn bản tiếng Trung sang thực thể Wikipedia tiếng Anh. Cần encoder đa ngữ hoặc một bước dịch.
- **KB cũ đi.** Công ty, sự kiện, con người mới không có trong dump Wikipedia năm ngoái. Pipeline production cần một vòng làm mới.

## Áp dụng

Stack của năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Tiếng Anh tổng quát + Wikipedia | BLINK hoặc REL |
| Đa ngữ, KB = Wikipedia | mGENRE |
| Thân thiện với LLM, ít mention/ngày | Prompt Claude/GPT-4 với danh sách ứng viên + JSON có ràng buộc |
| KB miền cụ thể (y tế, pháp lý) | BERT tuỳ biến với retrieval biết KB + fine-tune trên tập kiểu AIDA của miền |
| Độ trễ cực thấp | Chỉ prior khớp chính xác (baseline Milne-Witten) |
| SOTA nghiên cứu | GENRE / ExtEnD / LLM-EL sinh |

Mẫu production được ship năm 2026: NER → coref → EL trên từng mention → gộp cụm thành một thực thể chuẩn tắc cho mỗi cụm. Đầu ra: một KB id cho mỗi thực thể trong văn bản, không phải một id cho mỗi mention.

## Đóng gói

Lưu thành `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Bài tập

1. **Dễ.** Cài đặt bộ khử nhập nhằng prior+ngữ cảnh trong `code/main.py` trên 10 mention nhập nhằng (Paris, Jordan, Apple). Tự gán nhãn thực thể đúng. Đo độ chính xác.
2. **Trung bình.** Mã hoá 50 mention nhập nhằng bằng một sentence transformer. Embed mô tả của từng ứng viên. So sánh khử nhập nhằng dựa trên embedding với độ trùng lặp ngữ cảnh Jaccard.
3. **Khó.** Dựng một KB miền 1k thực thể (ví dụ: nhân viên + sản phẩm trong công ty bạn). Cài NER + EL end-to-end. Đo precision và recall trên 100 câu giữ lại.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Entity linking (EL) | Liên kết tới Wikipedia | Ánh xạ một mention về một mục KB duy nhất. |
| Candidate generation | Nó có thể là ai? | Trả về danh sách rút gọn các mục KB khả dĩ cho một mention. |
| Disambiguation | Chọn đúng cái | Chấm điểm ứng viên bằng ngữ cảnh, chọn cái thắng. |
| Alias index | Bảng tra cứu | Ánh xạ từ dạng bề mặt → thực thể ứng viên. |
| NIL | Không có trong KB | Dự đoán tường minh rằng không mục KB nào khớp. |
| KB | Cơ sở tri thức | Wikidata, Wikipedia, DBpedia, hoặc KB miền của bạn. |
| AIDA-CoNLL | Bộ benchmark | 1,393 bài Reuters với liên kết thực thể vàng. |

## Đọc thêm

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://researchcommons.waikato.ac.nz/entities/publication/b9a0b520-abc5-47c5-a86a-da6c579893ab) — cách tiếp cận prior+ngữ cảnh nền tảng.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — con ngựa thồ dựa trên embedding.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — EL sinh với constrained decoding.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — bài báo về benchmark.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — stack production mở.

---

[← Bài trước: Phân giải đồng tham chiếu (coreference resolution)](../../24-coreference-resolution/docs/vi.md) · [Bài sau: Trích xuất quan hệ & Xây dựng đồ thị tri thức →](../../26-relation-extraction-kg/docs/vi.md)
