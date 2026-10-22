# Phân giải đồng tham chiếu (coreference resolution)

> "She called him. He did not answer. The doctor was at lunch." Ba tham chiếu tới hai người mà không ai được nêu tên. Coreference resolution tìm ra ai là ai.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 06 (NER), Giai đoạn 5, Bài 07 (POS & phân tích cú pháp)
**Time:** ~60 phút

## Vấn đề

Trích xuất mọi đề cập (mention) đến Apple Inc. từ một bài báo 300 từ. Dễ khi bài viết "Apple". Khó khi bài viết "the company", "they", "Cupertino's technology giant" hay "Jobs's firm". Nếu không gộp các đề cập này về cùng một thực thể, pipeline NER của bạn sẽ bỏ sót 60-80% số đề cập.

Coreference resolution nối mọi biểu thức cùng chỉ về một thực thể trong thế giới thực thành một cụm (cluster). Nó là keo dính giữa NLP bề mặt (NER, phân tích cú pháp) và ngữ nghĩa phía sau (IE, QA, tóm tắt, KG).

Vì sao điều này quan trọng trong 2026:

- Tóm tắt: "The CEO announced..." so với "Tim Cook announced..." — bản tóm tắt nên nêu tên CEO.
- Hỏi đáp: "Who did she call?" đòi hỏi phân giải "she".
- Trích xuất thông tin: một knowledge graph có "PER1 founded Apple" và "Jobs founded Apple" thành hai mục riêng là sai.
- IE đa văn bản: gộp các đề cập giữa nhiều bài báo về cùng một sự kiện là coreference xuyên văn bản.

## Khái niệm

![Phân cụm đồng tham chiếu: đề cập → thực thể](../assets/coref.svg)

**Bài toán.** Đầu vào: một văn bản. Đầu ra: một phân cụm các đề cập (span) sao cho mỗi cụm chỉ về một thực thể.

**Các loại đề cập.**

- **Thực thể có tên.** "Tim Cook"
- **Danh từ chung.** "the CEO", "the company"
- **Đại từ (pronoun).** "he", "she", "they", "it"
- **Đồng vị.** "Tim Cook, Apple's CEO,"

**Các kiến trúc.**

1. **Dựa trên luật (Hobbs, 1978).** Phân giải đại từ dựa trên cây cú pháp với các luật ngữ pháp. Baseline tốt. Khó bị vượt qua một cách đáng ngạc nhiên ở mảng đại từ.
2. **Bộ phân loại cặp đề cập.** Với mỗi cặp đề cập (m_i, m_j), dự đoán xem chúng có đồng tham chiếu không. Gom cụm bằng bao đóng bắc cầu. Chuẩn trước 2016.
3. **Xếp hạng đề cập.** Với mỗi đề cập, xếp hạng các tiền ngữ ứng viên (kể cả "không có tiền ngữ"). Chọn cái đứng đầu.
4. **End-to-end dựa trên span (Lee et al., 2017).** Bộ mã hóa Transformer. Liệt kê mọi span ứng viên tới một giới hạn độ dài. Dự đoán điểm đề cập. Dự đoán xác suất tiền ngữ cho từng span. Gom cụm tham lam. Mặc định hiện đại.
5. **Sinh (2024+).** Prompt một LLM: "List every pronoun in this text and its antecedent." Chạy tốt ở các ca dễ, chật vật với văn bản dài và các thực thể hiếm gặp.

**Các thước đo đánh giá.** Năm thước đo chuẩn (MUC, B³, CEAF, BLANC, LEA) vì không thước đo nào một mình bắt được chất lượng phân cụm. Báo cáo trung bình của ba thước đo đầu dưới tên CoNLL F1. SOTA năm 2026 trên CoNLL-2012: ~83 F1.

**Các ca khó đã biết.**

- Mô tả xác định chỉ về thực thể được giới thiệu từ nhiều trang trước.
- Ẩn dụ cầu nối (bridging anaphora): "the wheels" → một chiếc xe đã nhắc trước đó.
- Đại từ ẩn (zero anaphora) trong các ngôn ngữ như tiếng Trung và tiếng Nhật.
- Cataphora (đại từ đứng trước người được chỉ): "When **she** walked in, Mary smiled."

```figure
coref-links
```

## Tự tay làm

### Bước 1: coreference nơ-ron huấn luyện sẵn (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Trên một văn bản dài hơn, bạn sẽ nhận được đại loại:
- Cụm 1: [Apple, The company, they]
- Cụm 2: [new products]

### Bước 2: bộ phân giải đại từ dựa trên luật (để học)

Xem `code/main.py` để có bản cài đặt chỉ dùng thư viện chuẩn:

1. Trích xuất đề cập: thực thể có tên (span viết hoa), đại từ (tra từ điển), mô tả xác định ("the X").
2. Với mỗi đại từ, xem K đề cập trước đó và cho điểm theo:
   - hợp giống/hợp số (heuristic)
   - độ gần đây (gần hơn thì thắng)
   - vai trò cú pháp (ưu tiên chủ ngữ)
3. Nối tiền ngữ (antecedent) có điểm cao nhất.

Không cạnh tranh được với mô hình nơ-ron. Nhưng nó cho thấy không gian tìm kiếm và những quyết định mà một mô hình end-to-end phải làm.

### Bước 3: dùng LLM cho coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Hai kiểu hỏng cần để ý. Thứ nhất, LLM gộp quá tay ("him" và "her" chỉ hai người khác nhau). Thứ hai, LLM âm thầm bỏ mất đề cập trong văn bản dài. Luôn kiểm chứng bằng cách soát offset của span.

### Bước 4: đánh giá

Script conll-2012 tiêu chuẩn tính MUC, B³, CEAF-φ4 và báo cáo trung bình. Với đánh giá nội bộ, hãy bắt đầu bằng precision và recall mức span trên tập test đã gán nhãn, rồi thêm F1 nối đề cập.

## Cạm bẫy

- **Bùng nổ singleton (đề cập đơn lẻ).** Một số hệ thống báo mỗi đề cập là một cụm riêng. B³ thì dễ dãi. MUC thì phạt nặng. Luôn kiểm cả ba thước đo.
- **Đại từ trong ngữ cảnh dài.** Hiệu năng giảm ~15 F1 trên văn bản quá 2.000 token. Chia khối cẩn thận.
- **Giả định về giống.** Luật giống cứng nhắc hỏng với người phi nhị phân, tổ chức, động vật. Hãy dùng mô hình học được hoặc cho điểm trung tính.
- **LLM trôi trên văn bản dài.** Một lần gọi API không thể gộp đề cập đáng tin qua 50+ đoạn. Dùng cửa sổ trượt + hợp nhất.

## Áp dụng

Bộ stack năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Tiếng Anh, một văn bản | `en_coreference_web_trf` (spaCy-experimental) hoặc AllenNLP neural coref |
| Đa ngôn ngữ | SpanBERT / XLM-R huấn luyện trên OntoNotes hoặc Multilingual CoNLL |
| Coref sự kiện xuyên văn bản | Mô hình end-to-end chuyên dụng (SOTA 2025–26) |
| Baseline LLM nhanh | GPT-4o / Claude với prompt coref đầu ra có cấu trúc |
| Hệ thống hội thoại production | Fallback dựa trên luật + mô hình nơ-ron chính + rà soát thủ công cho các slot quan trọng |

Mẫu tích hợp được đưa vào vận hành năm 2026: chạy NER trước, chạy coref, gộp cụm coref vào thực thể NER. Các tác vụ phía sau thấy một thực thể mỗi cụm, không phải một thực thể mỗi đề cập.

## Đóng gói

Lưu thành `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Bài tập

1. **Dễ.** Chạy bộ phân giải dựa trên luật trong `code/main.py` trên 5 đoạn tự viết. Đo độ chính xác nối đề cập so với ground truth.
2. **Trung bình.** Dùng một mô hình coref nơ-ron huấn luyện sẵn trên một bài báo. So sánh cụm với gán nhãn thủ công của bạn. Nó hỏng ở đâu?
3. **Khó.** Xây pipeline NER tăng cường coref: NER trước, rồi gộp qua cụm coref. Đo mức cải thiện độ phủ thực thể so với chỉ NER trên 100 bài báo.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Mention | Một tham chiếu | Một span văn bản chỉ về một thực thể (tên, đại từ, cụm danh từ). |
| Antecedent | Cái mà "it" chỉ tới | Đề cập trước đó mà đề cập sau đồng tham chiếu với nó. |
| Cluster | Các đề cập của thực thể | Tập đề cập cùng chỉ về một thực thể trong thế giới thực. |
| Anaphora | Tham chiếu ngược | Đề cập sau chỉ về đề cập trước ("he" → "John"). |
| Cataphora | Tham chiếu xuôi | Đề cập trước chỉ về đề cập sau ("When he arrived, John..."). |
| Bridging | Tham chiếu ngầm | "I bought a car. The wheels were bad." (bánh xe của CHIẾC xe đó.) |
| CoNLL F1 | Con số trên bảng xếp hạng | Trung bình điểm F1 của MUC, B³, CEAF-φ4. |

## Đọc thêm

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — chương sách giáo khoa kinh điển.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — end-to-end dựa trên span.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — pretraining cải thiện coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — bộ benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — kinh điển dựa trên luật.

---

[← Bài trước: Chiến lược chia khối (chunking) cho RAG](../../23-chunking-strategies-rag/docs/vi.md) · [Bài sau: Liên kết thực thể & Khử nhập nhằng →](../../25-entity-linking/docs/vi.md)
