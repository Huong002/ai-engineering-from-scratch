# Bag of Words, TF-IDF, và Biểu diễn Văn bản

> Đếm trước, nghĩ sau. Năm 2026, TF-IDF vẫn thắng embedding trên những bài toán được định nghĩa rõ ràng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 01 (Xử lý văn bản), Giai đoạn 2, Bài 02 (Hồi quy tuyến tính from scratch)
**Time:** ~75 phút

## Vấn đề

Mô hình cần số. Bạn thì có chuỗi.

Mọi pipeline NLP đều phải trả lời cùng một câu hỏi. Làm sao biến một dòng token có độ dài thay đổi thành một vector kích thước cố định mà bộ phân loại có thể dùng. Câu trả lời đầu tiên mà cả ngành chọn là cách ngớ ngẩn nhất mà vẫn chạy được. Đếm các từ. Tạo một vector.

Vector đó đã gánh nhiều NLP production hơn bất kỳ mô hình embedding nào. Bộ lọc spam, bộ phân loại chủ đề, phát hiện bất thường trong log, xếp hạng tìm kiếm (trước BM25), làn sóng phân tích cảm xúc đầu tiên, thập kỷ đầu của các benchmark NLP học thuật. Năm 2026, người làm thực tế vẫn với tay tới nó trước tiên cho những bài phân loại hẹp. Nó nhanh, giải thích được, và thường không phân biệt nổi với một mô hình embedding 400M tham số trong những bài mà sự xuất hiện của từ là điều quan trọng.

Bài này tự tay xây dựng bag of words (túi từ), rồi TF-IDF, từ đầu. Sau đó cho thấy scikit-learn làm đúng việc đó trong ba dòng. Rồi gọi tên chế độ thất bại khiến bạn phải tìm đến embedding.

## Khái niệm

**Bag of Words (BoW)** vứt bỏ thứ tự. Với mỗi tài liệu, đếm xem mỗi từ trong từ vựng (vocabulary) xuất hiện bao nhiêu lần. Độ dài vector là kích thước từ vựng. Vị trí `i` là số đếm của từ `i`.

**TF-IDF** đánh trọng số lại BoW. Một từ xuất hiện trong mọi tài liệu là từ không mang thông tin, nên giảm trọng số của nó xuống. Một từ hiếm trong toàn bộ kho văn bản nhưng lại nhiều trong một tài liệu là tín hiệu, nên tăng trọng số của nó lên.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Trong đó `TF` là term frequency (tần suất từ) trong tài liệu, `df` là document frequency (tần suất tài liệu — số tài liệu chứa từ đó), `N` là tổng số tài liệu. Hàm `log` giữ trọng số có chặn trên cho những từ xuất hiện ở khắp nơi.

Thuộc tính then chốt: cả hai đều tạo ra sparse vector (vector thưa) với các trục giải thích được. Bạn có thể nhìn trọng số của một bộ phân loại đã huấn luyện và đọc xem từ nào đẩy tài liệu về phía mỗi lớp. Bạn không làm được điều này với một embedding BERT 768 chiều.

```figure
bow-tfidf
```

## Tự tay làm

### Bước 1: xây dựng từ vựng

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Input: danh sách tài liệu đã tách token (tokenizer mức từ nào cũng được; `code/main.py` trong bài này dùng một biến thể đơn giản hoá, chữ thường). Output: dict `{word: index}`. Thứ tự chèn ổn định nghĩa là từ có index 0 là từ đầu tiên xuất hiện trong tài liệu đầu tiên. Quy ước mỗi nơi mỗi khác; scikit-learn sắp xếp theo bảng chữ cái.

### Bước 2: túi từ

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Mỗi hàng là một tài liệu. Mỗi cột là một chỉ số trong từ vựng. Phần tử `[i][j]` là "từ `j` xuất hiện bao nhiêu lần trong tài liệu `i`." Tài liệu 1 có `cat` hai lần vì nó có hai lần. Tài liệu 0 có `ran` không lần vì nó không có.

### Bước 3: tần suất từ và tần suất tài liệu

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Hai mẹo làm trơn đáng để gọi tên. `(n+1)/(d+1)` tránh `log(x/0)`. Phần `+1` ở cuối bảo đảm một từ xuất hiện trong mọi tài liệu vẫn có IDF bằng 1 (không phải 0), khớp với mặc định của scikit-learn. Các cài đặt khác dùng `log(N/df)` thô. Cả hai đều chạy; bản làm trơn thân thiện hơn.

### Bước 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Ba tài liệu, năm từ trong từ vựng (`the`, `cat`, `sat`, `dog`, `ran`). `the` xuất hiện trong cả ba, nên IDF của nó thấp. `dog` xuất hiện trong một, nên IDF của nó cao. Các vector là thưa (hầu hết phần tử đều nhỏ) và những từ phân biệt nổi bật lên.

### Bước 5: chuẩn hoá L2 theo từng hàng

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Không chuẩn hoá thì tài liệu dài hơn có vector lớn hơn và chi phối điểm tương đồng. Chuẩn hoá L2 đưa mọi tài liệu lên mặt siêu cầu đơn vị. Cosine similarity (độ tương đồng cosine) giữa các hàng giờ chỉ còn là một tích vô hướng.

## Áp dụng

scikit-learn có sẵn bản dùng cho production.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` làm tokenization, từ vựng, và BoW trong một lần gọi. `TfidfVectorizer` thêm đánh trọng số IDF và chuẩn hoá L2. Cả hai trả về ma trận thưa. Với 100k tài liệu, bản dense không vừa bộ nhớ; hãy giữ sparse cho tới khi bộ phân loại đòi dense.

Những núm vặn thay đổi mọi thứ:

| Arg | Tác dụng |
|-----|--------|
| `ngram_range=(1, 2)` | Bao gồm bigram. Thường tăng điểm phân loại. |
| `min_df=2` | Bỏ những từ xuất hiện trong ít hơn 2 tài liệu. Cắt bớt từ vựng trên dữ liệu nhiễu. |
| `max_df=0.95` | Bỏ những từ xuất hiện trong hơn 95% tài liệu. Xấp xỉ việc loại stopword mà không cần danh sách cứng. |
| `stop_words="english"` | Danh sách stopword có sẵn của scikit-learn. Tuỳ bài toán — phân tích cảm xúc *không* nên bỏ phủ định. |
| `sublinear_tf=True` | Dùng `1 + log(tf)` thay vì `tf` thô. Có ích khi một từ lặp nhiều lần trong một tài liệu. |

### Khi nào TF-IDF vẫn thắng (tính đến 2026)

- Phát hiện spam, gán nhãn chủ đề, đánh dấu bất thường trong log. Sự xuất hiện của từ là điều quan trọng; sắc thái ngữ nghĩa thì không.
- Chế độ ít dữ liệu (hàng trăm ví dụ có nhãn). TF-IDF cộng hồi quy logistic không tốn chi phí tiền huấn luyện.
- Bất cứ đâu độ trễ quan trọng. TF-IDF cộng một mô hình tuyến tính trả lời trong micro giây. Embedding một tài liệu qua transformer mất 10-100ms.
- Những hệ thống buộc phải giải thích dự đoán của mình. Xem hệ số của bộ phân loại. Những từ có hệ số dương cao nhất chính là lý do.

### Khi nào TF-IDF thất bại

Thất bại vì mù ngữ nghĩa. Xét hai tài liệu này:

- "The movie was not good at all."
- "The movie was excellent."

Một cái là đánh giá tiêu cực. Một cái là tích cực. Phần trùng TF-IDF của chúng đúng bằng `{the, movie, was}`. Một bộ phân loại bag of words phải học thuộc rằng từ `not` đứng gần `good` sẽ lật nhãn. Nó có thể học được điều này nếu đủ dữ liệu, nhưng không bao giờ mượt như một mô hình hiểu cú pháp.

Thất bại còn lại: từ ngoài từ vựng tại thời điểm suy luận. Một mô hình BoW huấn luyện trên đánh giá IMDb không biết làm gì với `Zoomer-approved` nếu token đó chưa từng xuất hiện khi huấn luyện. Embedding mức subword (bài 04) xử lý được việc này. TF-IDF thì không.

### Kết hợp: embedding đánh trọng số bằng TF-IDF

Mặc định thực dụng của năm 2026 cho phân loại dữ liệu cỡ trung bình: dùng trọng số TF-IDF như attention trên embedding của từ.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Bạn có năng lực ngữ nghĩa từ embedding, và sự nhấn mạnh từ hiếm từ TF-IDF. Bộ phân loại huấn luyện trên vector đã gộp. Cách này vượt cả hai phương pháp khi dùng riêng cho phân loại cảm xúc, chủ đề và ý định dưới khoảng 50k ví dụ có nhãn.

## Đóng gói

Lưu thành `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Bài tập

1. **Dễ.** Cài đặt `cosine_similarity(doc_vec_a, doc_vec_b)` trên đầu ra TF-IDF đã chuẩn hoá L2. Kiểm chứng rằng hai tài liệu giống hệt cho điểm 1.0 và hai tài liệu có từ vựng rời nhau cho điểm 0.0.
2. **Trung bình.** Thêm hỗ trợ n-gram (n-gram) vào `bag_of_words`. Tham số `n` tạo ra số đếm trên các `n`-gram. Kiểm tra rằng `n=2` trên `["the", "cat", "sat"]` tạo ra số đếm bigram cho `["the cat", "cat sat"]`.
3. **Khó.** Xây dựng bản kết hợp TF-IDF đánh trọng số embedding ở trên bằng vector GloVe 100d (tải một lần, cache). So sánh độ chính xác phân loại với TF-IDF thuần và embedding mean-pooled thuần trên bộ dữ liệu 20 Newsgroups. Báo cáo phương pháp nào thắng ở đâu.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| BoW | Vector tần suất từ | Số đếm các từ trong từ vựng trong một tài liệu. Vứt bỏ thứ tự. |
| TF | Term frequency | Số lần một từ xuất hiện trong tài liệu, có thể chuẩn hoá theo độ dài tài liệu. |
| DF | Document frequency | Số tài liệu chứa từ đó ít nhất một lần. |
| IDF | Inverse document frequency (nghịch đảo tần suất tài liệu) | `log(N / df)` có làm trơn. Giảm trọng số những từ xuất hiện ở khắp nơi. |
| Sparse vector | Phần lớn là zero | Từ vựng thường 10k-100k từ; hầu hết đều vắng mặt trong một tài liệu bất kỳ. |
| Cosine similarity | Góc giữa hai vector | Tích vô hướng của các vector đã chuẩn hoá L2. 1 là trùng nhau, 0 là trực giao. |

## Đọc thêm

- [scikit-learn — trích xuất đặc trưng từ văn bản](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — tài liệu tham chiếu API chuẩn, kèm ghi chú về từng núm vặn.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — bài báo đưa TF-IDF thành mặc định suốt một thập kỷ.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — góc nhìn năm 2026 về khi nào phương pháp cũ thắng và vì sao.

---

[← Bài trước: Xử lý văn bản](../../01-text-processing/docs/vi.md) · [Bài sau: Word Embeddings →](../../03-word-embeddings-word2vec/docs/vi.md)
