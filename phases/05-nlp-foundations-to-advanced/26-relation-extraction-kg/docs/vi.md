# Trích xuất quan hệ & Xây dựng đồ thị tri thức

> NER đã tìm ra các thực thể. Liên kết thực thể đã neo chúng. Trích xuất quan hệ (relation extraction) tìm ra các cạnh giữa chúng. Đồ thị tri thức (knowledge graph) là tổng của các nút, các cạnh và nguồn gốc của chúng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 06 (NER), Giai đoạn 5, Bài 25 (Liên kết thực thể)
**Time:** ~60 phút

## Vấn đề

Một nhà phân tích đọc: "Tim Cook became CEO of Apple in 2011." Bốn sự thật:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Trích xuất quan hệ (RE) biến văn bản tự do thành các bộ ba (triple) có cấu trúc `(subject, relation, object)`. Tổng hợp trên cả một kho văn bản là bạn có một đồ thị tri thức. Tổng hợp rồi truy vấn là bạn có một nền tảng suy luận cho RAG, phân tích, hoặc kiểm toán tuân thủ.

Vấn đề của năm 2026: LLM trích xuất quan hệ rất hào hứng. Hào hứng quá mức. Chúng tạo ra những bộ ba mà văn bản nguồn không hề chống lưng. Không có nguồn gốc, bạn không thể phân biệt bộ ba thật với hư cấu nghe hợp lý. Câu trả lời của năm 2026 là các pipeline kiểu AEVS: neo rồi kiểm chứng.

## Khái niệm

![Văn bản → bộ ba → đồ thị tri thức](../assets/relation-extraction.svg)

**Dạng bộ ba.** `(subject_entity, relation_type, object_entity)`. Quan hệ đến từ một ontology đóng (thuộc tính Wikidata, FIBO, UMLS) hoặc một tập mở (kiểu OpenIE, gì cũng được).

**Ba cách trích xuất.**

1. **Dựa trên luật / mẫu.** Mẫu Hearst: "X such as Y" → `(Y, isA, X)`. Cộng thêm regex viết tay. Dễ vỡ, chính xác, giải thích được.
2. **Bộ phân loại có giám sát.** Cho hai lần nhắc thực thể trong một câu, dự đoán quan hệ từ một tập cố định. Huấn luyện trên TACRED, ACE, KBP. Chuẩn mực giai đoạn 2015–2022.
3. **LLM sinh.** Prompt mô hình phát ra các bộ ba. Chạy được ngay. Phải có nguồn gốc, nếu không sẽ sinh ra rác trông có lý.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Khung giảm thiểu ảo giác hiện hành:

- **Neo.** Xác định mọi span thực thể và span cụm quan hệ với vị trí chính xác.
- **Trích xuất.** Sinh các bộ ba gắn với các span neo.
- **Kiểm chứng.** Đối chiếu từng thành phần của bộ ba với văn bản nguồn; loại bỏ mọi thứ không được chống lưng.
- **Bổ sung.** Một lượt phủ sóng bảo đảm không span neo nào bị bỏ sót.

Ảo giác giảm mạnh. Cần thêm tính toán nhưng kiểm toán được.

**Đánh đổi đóng-vs-mở.**

- **Ontology đóng.** Danh sách thuộc tính cố định (ví dụ 11.000+ thuộc tính của Wikidata). Dự đoán được. Truy vấn được. Khó bịa.
- **Open IE.** Mọi cụm động từ đều thành quan hệ. Độ phủ (recall) cao. Độ chính xác (precision) thấp. Truy vấn lộn xộn.

KG trong sản phẩm thường pha trộn: Open IE để khám phá, rồi chuẩn hoá quan hệ về một ontology đóng trước khi hợp nhất vào đồ thị chính.

```figure
relation-triples
```

## Tự tay làm

### Bước 1: trích xuất dựa trên mẫu

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Xem `code/main.py` cho bộ trích xuất đồ chơi đầy đủ. Mẫu Hearst vẫn có mặt trong các pipeline chuyên ngành vì chúng dễ gỡ lỗi.

### Bước 2: phân loại quan hệ có giám sát

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL là một bộ trích xuất quan hệ seq2seq: văn bản vào, bộ ba ra, đã ở dạng id thuộc tính Wikidata. Được fine-tune trên dữ liệu giám sát từ xa (distant supervision). Là baseline mở trọng số tiêu chuẩn.

### Bước 3: trích xuất bằng prompt LLM có neo

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Kiểm chứng từng span trả về với nguồn. Loại bỏ mọi thứ mà `text[start:end] != triple_entity`. Đây là bước "verify" của AEVS ở dạng tối giản.

### Bước 4: chuẩn hoá về một ontology đóng

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Chuẩn hoá thường chiếm 60-80% công việc kỹ thuật. Hãy tính trước ngân sách cho nó.

### Bước 5: xây một đồ thị nhỏ và truy vấn

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Đây là nguyên tử của mọi hệ RAG-over-KG. Mở rộng nó bằng kho bộ ba RDF (Blazegraph, Virtuoso), đồ thị thuộc tính (Neo4j), hoặc kho đồ thị tăng cường vector.

## Cạm bẫy

- **Coreference trước RE.** "He founded Apple" — RE cần biết "he" là ai. Chạy coref trước (bài 24).
- **Chuẩn hoá thực thể.** "Apple Inc" và "Apple" phải quy về cùng một nút. Liên kết thực thể trước (bài 25).
- **Bộ ba ảo giác.** LLM phát ra bộ ba mà văn bản không chống lưng. Hãy bắt buộc kiểm chứng span.
- **Trôi chuẩn hoá quan hệ.** Quan hệ Open IE không nhất quán ("was born in," "came from," "is a native of"). Gộp về id chuẩn, nếu không đồ thị không truy vấn được.
- **Lỗi thời gian.** "Tim Cook is CEO of Apple" — đúng hiện nay, sai vào năm 2005. Nhiều quan hệ bị giới hạn thời gian. Dùng qualifier (`P580` thời điểm bắt đầu, `P582` thời điểm kết thúc trong Wikidata).
- **Lệch miền.** REBEL huấn luyện trên Wikipedia. Văn bản pháp lý, y khoa và khoa học thường cần mô hình RE fine-tune theo miền.

## Áp dụng

Stack của năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Sản xuất nhanh, miền tổng quát | REBEL hoặc LlamaPred kèm chuẩn hoá Wikidata |
| Chuyên ngành (biomed, pháp lý) | Fine-tune theo miền kiểu SciREX + ontology tuỳ chỉnh |
| Dùng prompt LLM, đầu ra được kiểm toán | Pipeline AEVS: neo → trích xuất → kiểm chứng → bổ sung |
| Trích xuất tin tức khối lượng lớn | Kết hợp mẫu + có giám sát |
| Xây KG từ đầu | Open IE + một lượt chuẩn hoá thủ công |
| KG thời gian | Trích xuất kèm qualifier (thời gian bắt đầu/kết thúc, thời điểm) |

Mẫu tích hợp: NER → coref → liên kết thực thể → trích xuất quan hệ → ánh xạ ontology → nạp vào đồ thị. Mỗi giai đoạn là một cổng chất lượng tiềm năng.

## Đóng gói

Lưu thành `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Bài tập

1. **Dễ.** Chạy bộ trích xuất mẫu trong `code/main.py` trên 5 câu từ bài báo. Tự kiểm tra độ chính xác.
2. **Trung bình.** Dùng REBEL (hoặc một LLM nhỏ) trên cùng các câu đó. So sánh bộ ba. Bộ trích xuất nào có độ chính xác cao hơn? Độ phủ cao hơn?
3. **Khó.** Xây pipeline AEVS: trích xuất bằng LLM + kiểm chứng span với nguồn. Đo tỉ lệ ảo giác trước và sau bước kiểm chứng trên 50 câu kiểu Wikipedia.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | Bộ ba `(s, r, o)` là đơn vị nguyên tử của một KG. |
| Open IE | Trích xuất mọi thứ | Cụm quan hệ từ vựng mở; độ phủ cao, độ chính xác thấp. |
| Closed ontology | Lược đồ cố định | Tập quan hệ hữu hạn (Wikidata, UMLS, FIBO). |
| Canonicalization | Chuẩn hoá mọi thứ | Ánh xạ tên bề mặt / quan hệ về id chuẩn. |
| AEVS | Trích xuất có neo | Pipeline Anchor-Extraction-Verification-Supplement (2026). |
| Provenance | Liên kết nguồn gốc | Mỗi bộ ba mang doc id + span ký tự tới nguồn của nó. |
| Distant supervision | Nhãn giá rẻ | Căn chỉnh văn bản với một KG có sẵn để tạo dữ liệu huấn luyện. |

## Đọc thêm

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — bài báo về distant supervision.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — con ngựa thồ seq2seq cho RE.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — trích xuất thông tin kết hợp.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — thiết kế giảm thiểu ảo giác năm 2026.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — truy vấn đồ thị chuẩn.

---

[← Bài trước: Liên kết thực thể & Khử nhập nhằng](../../25-entity-linking/docs/vi.md) · [Bài sau: Đánh giá LLM →](../../27-llm-evaluation-frameworks/docs/vi.md)
