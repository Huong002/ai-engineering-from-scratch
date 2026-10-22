# Mô hình hoá chủ đề — LDA và BERTopic

> LDA: tài liệu là hỗn hợp các chủ đề, chủ đề là phân phối trên các từ. BERTopic: tài liệu tạo cụm trong không gian embedding, cụm chính là chủ đề. Cùng một mục tiêu, cách phân rã khác nhau.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 02 (BoW + TF-IDF), Giai đoạn 5, Bài 03 (Word2Vec)
**Time:** ~45 phút

## Vấn đề

Bạn có 10.000 ticket hỗ trợ khách hàng, 50.000 bài báo, hoặc 200.000 tweet. Bạn cần biết cả tập dữ liệu đang nói về cái gì mà không phải đọc hết. Bạn không có nhãn phân loại. Bạn thậm chí không biết có bao nhiêu nhãn tồn tại.

topic modeling (mô hình hoá chủ đề) trả lời câu hỏi đó mà không cần giám sát. Đưa vào một corpus, nhận lại một tập nhỏ chủ đề mạch lạc và, với mỗi tài liệu, một phân phối trên các chủ đề đó.

Hai họ thuật toán chiếm ưu thế. LDA (2003) coi mỗi tài liệu là hỗn hợp của các chủ đề latent (ẩn) và mỗi chủ đề là một phân phối trên các từ. Suy luận là Bayesian. Nó vẫn được đưa vào production khi bạn cần gán chủ đề kiểu mixed-membership và các phân phối xác suất theo từ có thể giải thích được.

BERTopic (2020) mã hoá tài liệu bằng BERT, giảm chiều bằng UMAP, gom cụm bằng HDBSCAN, và trích từ khoá chủ đề qua class-based TF-IDF. Nó thắng ở văn bản ngắn, mạng xã hội, và bất cứ đâu mà độ tương đồng ngữ nghĩa quan trọng hơn việc trùng từ. Một tài liệu nhận một chủ đề, đó là hạn chế với nội dung dài.

Bài này xây trực giác cho cả hai và chỉ rõ nên chọn cái nào cho một corpus cụ thể.

## Khái niệm

![Mô hình hỗn hợp LDA so với cách gom cụm của BERTopic](../assets/topic-modeling.svg)

**Câu chuyện sinh dữ liệu của LDA.** Mỗi chủ đề là một phân phối trên các từ. Mỗi tài liệu là hỗn hợp các chủ đề. Để sinh một từ trong tài liệu, lấy mẫu một chủ đề từ hỗn hợp của tài liệu, rồi lấy mẫu một từ từ phân phối của chủ đề đó. Suy luận đi ngược lại: cho các từ quan sát được, suy ra phân phối chủ đề (topic distribution) cho từng tài liệu và phân phối từ cho từng chủ đề. Collapsed Gibbs sampling (lấy mẫu Gibbs thu gọn) hoặc variational Bayes lo phần toán học.

Đầu ra chính của LDA:

- `doc_topic`: ma trận văn bản–chủ đề (document-topic matrix) `(n_docs, n_topics)`, mỗi dòng cộng lại bằng 1 (hỗn hợp chủ đề của tài liệu).
- `topic_word`: ma trận `(n_topics, vocab_size)`, mỗi dòng cộng lại bằng 1 (phân phối từ của chủ đề).

**Pipeline của BERTopic.**

1. Mã hoá mỗi tài liệu bằng sentence transformer (ví dụ `all-MiniLM-L6-v2`). Vector 384 chiều.
2. Giảm chiều bằng UMAP xuống còn ~5 chiều. Embedding của BERT quá nhiều chiều để gom cụm.
3. Gom cụm bằng HDBSCAN. Dựa trên mật độ, tạo ra các cụm kích thước khác nhau và một nhãn "outlier".
4. Với mỗi cụm, tính class-based TF-IDF trên các tài liệu của cụm để trích ra các từ hàng đầu.

Đầu ra là một chủ đề cho mỗi tài liệu (cộng nhãn outlier -1). Tuỳ chọn, có thể có membership mềm qua vector xác suất của HDBSCAN.

```figure
topic-drift
```

## Tự tay làm

### Bước 1: LDA qua scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Lưu ý: stopword đã bị loại, min_df và max_df lọc các từ hiếm và từ xuất hiện ở khắp nơi, dùng CountVectorizer (không phải TfidfVectorizer) vì LDA cần số đếm thô.

### Bước 2: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Bộ lọc `Topic != -1` loại bỏ nhóm outlier của BERTopic (những tài liệu HDBSCAN không gom cụm được). `min_topic_size` điều khiển kích thước cụm tối thiểu của HDBSCAN; mặc định của thư viện BERTopic là 10. Ví dụ này đặt 15 cho vừa quy mô của bài. Với corpus trên 10.000 tài liệu, hãy tăng lên 50 hoặc 100.

### Bước 3: đánh giá

Cả hai phương pháp đều cho ra từ khoá chủ đề. Câu hỏi là các từ đó có mạch lạc với nhau không.

- **Độ mạch lạc chủ đề (topic coherence, c_v).** Kết hợp NPMI (normalized pointwise mutual information) của các cặp từ hàng đầu trong ngữ cảnh cửa sổ trượt, tổng hợp điểm thành vector chủ đề, rồi so sánh các vector đó bằng cosine similarity. Càng cao càng tốt. Dùng `gensim.models.CoherenceModel` với `coherence="c_v"`.
- **Độ đa dạng chủ đề.** Tỉ lệ từ duy nhất trên tổng các từ hàng đầu của tất cả chủ đề. Càng cao càng tốt (các chủ đề không chồng lấn nhau).
- **Kiểm tra định tính.** Đọc các từ hàng đầu của mỗi chủ đề. Chúng có gọi tên một thứ có thật không? Phán đoán của con người vẫn là chốt chặn cuối cùng.

## Khi nào chọn cái nào

| Tình huống | Nên chọn |
|-----------|------|
| Văn bản ngắn (tweet, review, tiêu đề) | BERTopic |
| Tài liệu dài có hỗn hợp chủ đề | LDA |
| Không có GPU / ít compute | LDA hoặc NMF |
| Cần phân phối nhiều chủ đề ở mức tài liệu | LDA |
| Tích hợp LLM để đặt tên chủ đề | BERTopic (hỗ trợ sẵn) |
| Triển khai edge với tài nguyên hạn chế | LDA |
| Độ mạch lạc ngữ nghĩa tối đa | BERTopic |

Yếu tố thực tế lớn nhất là độ dài tài liệu. Embedding BERT bị cắt bớt; cách đếm của LDA hoạt động với độ dài nào cũng được. Với tài liệu dài hơn cửa sổ ngữ cảnh của model embedding, hoặc chia nhỏ rồi tổng hợp, hoặc dùng LDA.

## Áp dụng

Bộ công cụ năm 2026:

- **BERTopic.** Mặc định cho văn bản ngắn và mọi thứ mà ngữ nghĩa quan trọng.
- **`gensim.models.LdaModel`.** LDA kinh điển cho production, chín muồi, đã được kiểm chứng thực tế.
- **`sklearn.decomposition.LatentDirichletAllocation`.** LDA dễ dùng cho thử nghiệm.
- **NMF.** Phân rã ma trận không âm. Thay thế nhanh cho LDA, chất lượng tương đương trên văn bản ngắn.
- **Top2Vec.** Thiết kế tương tự BERTopic. Cộng đồng nhỏ hơn nhưng tốt trên một số benchmark.
- **FASTopic.** Mới hơn, nhanh hơn BERTopic trên corpus rất lớn.
- **Đặt tên bằng LLM.** Chạy bất kỳ cách gom cụm nào, rồi prompt một model để đặt tên cho từng cụm.

## Đóng gói

Lưu thành `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Bài tập

1. **Dễ.** Fit LDA với 5 chủ đề trên bộ dữ liệu 20 Newsgroups. In 10 từ hàng đầu của mỗi chủ đề. Tự gán nhãn từng chủ đề bằng tay. Thuật toán có tìm ra đúng các hạng mục thật không?
2. **Trung bình.** Fit BERTopic trên cùng tập con 20 Newsgroups. So sánh số chủ đề tìm được, các từ hàng đầu, và độ mạch lạc định tính với LDA. Cách nào làm nổi các hạng mục thật rõ ràng hơn?
3. **Khó.** Tính c_v coherence cho cả LDA và BERTopic trên corpus của bạn. Chạy mỗi cách với 5, 10, 20, 50 chủ đề. Vẽ đồ thị coherence theo số chủ đề. Báo cáo phương pháp nào ổn định hơn khi số chủ đề thay đổi.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Topic | Một thứ mà corpus nói về | Một phân phối xác suất trên các từ (LDA) hoặc một cụm các tài liệu tương tự (BERTopic). |
| Mixed membership | Tài liệu thuộc nhiều chủ đề | LDA gán cho mỗi tài liệu một phân phối trên tất cả các chủ đề. |
| UMAP | Giảm chiều | Học đa tạp giữ cấu trúc cục bộ; dùng trong BERTopic. |
| HDBSCAN | Gom cụm theo mật độ | Tìm các cụm kích thước thay đổi; gán nhãn "noise" (-1) cho outlier. |
| c_v coherence | Thước đo chất lượng chủ đề | Trung bình pointwise mutual information của các từ hàng đầu trong các cửa sổ trượt. |

## Đọc thêm

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — bài báo LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — bài báo BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — bài báo giới thiệu c_v và các metric cùng họ.
- [Tài liệu BERTopic](https://maartengr.github.io/BERTopic/) — tài liệu tham chiếu cho production. Ví dụ rất tốt.

---

[← Bài trước: Truy hồi thông tin (information retrieval) và Tìm kiếm](../../14-information-retrieval-search/docs/vi.md) · [Bài sau: Sinh văn bản trước Transformer →](../../16-text-generation-pre-transformer/docs/vi.md)
