# Word Embeddings — Word2Vec từ đầu

> Một từ được biết qua những từ nó đi cùng. Huấn luyện một mạng nông dựa trên ý tưởng đó và hình học tự hiện ra.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 02 (BoW + TF-IDF), Giai đoạn 3, Bài 03 (Lan truyền ngược from scratch)
**Time:** ~75 phút

## Vấn đề

TF-IDF biết `dog` và `puppy` là hai từ khác nhau. Nó không biết hai từ này gần như cùng nghĩa. Một bộ phân loại huấn luyện trên `dog` không thể tổng quát hoá sang một đánh giá nói về `puppy`. Bạn có thể vá tạm bằng cách liệt kê từ đồng nghĩa, nhưng cách đó thất bại với thuật ngữ hiếm, tiếng lóng chuyên ngành, và mọi ngôn ngữ bạn không lường trước.

Bạn muốn một biểu diễn trong đó `dog` và `puppy` nằm gần nhau trong không gian. Trong đó `king - man + woman` rơi gần `queen`. Trong đó một mô hình huấn luyện trên `dog` chuyển một phần tín hiệu sang `puppy` miễn phí.

Word2Vec đã cho chúng ta không gian đó. Mạng nơ-ron hai lớp, những lần huấn luyện hàng nghìn tỷ token, công bố năm 2013. Kiến trúc đơn giản đến mức gần như ngượng ngùng. Kết quả đã định hình lại NLP trong một thập kỷ.

## Khái niệm

**Giả thuyết phân phối** (Distributional hypothesis, Firth, 1957): "Bạn sẽ biết một từ qua những từ nó đi cùng." Nếu hai từ xuất hiện trong ngữ cảnh giống nhau, chúng có lẽ mang nghĩa giống nhau.

Word2Vec có hai phiên bản, cả hai đều khai thác ý tưởng đó.

- **Skip-gram (dự đoán ngữ cảnh từ từ trung tâm).** Cho một từ trung tâm, dự đoán các từ xung quanh. `cat -> (the, sat, on)` với cửa sổ ngữ cảnh (context window) kích thước 2.
- **CBOW (continuous bag of words).** Cho các từ xung quanh, dự đoán từ trung tâm. `(the, sat, on) -> cat`.

Skip-gram huấn luyện chậm hơn nhưng xử lý từ hiếm tốt hơn. Nó đã trở thành mặc định.

Mạng chỉ có một lớp ẩn, không có phi tuyến. Đầu vào là một vector one-hot trên toàn bộ từ vựng. Đầu ra là softmax trên toàn bộ từ vựng. Sau khi huấn luyện, bạn vứt lớp đầu ra đi. Trọng số lớp ẩn chính là embedding từ (word embedding).

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Mẹo ở đây: softmax trên 100k từ là quá đắt. Word2Vec dùng lấy mẫu âm (negative sampling) để biến nó thành bài toán phân loại nhị phân. Dự đoán "từ ngữ cảnh này có xuất hiện gần từ trung tâm này không, có hay không". Lấy một nhúm từ âm (không đồng xuất hiện) cho mỗi cặp huấn luyện thay vì tính softmax trên toàn bộ từ vựng.

```figure
word-vector-arithmetic
```

## Tự tay làm

### Bước 1: các cặp huấn luyện từ một corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Mỗi cặp (từ trung tâm, ngữ cảnh) trong một cửa sổ là một ví dụ huấn luyện dương.

### Bước 2: bảng embedding (ma trận embedding)

Hai ma trận. `W` là bảng embedding của từ trung tâm (bảng bạn giữ lại). `W'` là bảng của từ ngữ cảnh (thường bị bỏ đi, đôi khi được lấy trung bình với `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Khởi tạo ngẫu nhiên nhỏ. Vocab 10k và dim 100 là thực tế; để dạy học, 50 từ vựng × 16 chiều là đủ để thấy hình học.

### Bước 3: mục tiêu lấy mẫu âm

Với mỗi cặp dương `(center, context)`, lấy mẫu `k` từ ngẫu nhiên trong từ vựng làm từ âm. Huấn luyện mô hình sao cho tích vô hướng `W[center] · W'[context]` cao với cặp dương và thấp với cặp âm.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Công thức kỳ diệu: mất mát logistic trên cặp dương (muốn sigmoid gần 1) cộng mất mát logistic trên các cặp âm (muốn sigmoid gần 0). Gradient chảy về cả hai bảng (đây chính là hạ gradient, gradient descent, bằng tay). Bản suy dẫn đầy đủ nằm trong bài báo gốc; hãy đi qua nó một lần với bút và giấy nếu bạn muốn nhớ lâu.

### Bước 4: huấn luyện trên một corpus đồ chơi

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Sau đủ số epoch trên một corpus lớn, những từ chia sẻ ngữ cảnh có embedding trung tâm giống nhau. Trên corpus đồ chơi, bạn thấy hiệu ứng mờ nhạt. Trên hàng tỷ token, bạn thấy nó rõ rệt.

### Bước 5: mẹo loại suy

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Trên vector Google News 300 chiều đã huấn luyện sẵn:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Không phải vì mô hình biết hoàng tộc là gì. Mà vì vector `(king - man)` nắm bắt được một thứ đại loại như "thuộc hoàng gia", và cộng nó vào `woman` thì rơi gần vùng "nữ hoàng gia".

## Áp dụng

Viết Word2Vec từ đầu là để dạy học. NLP trong sản xuất dùng `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Với công việc thật, gần như bạn không bao giờ tự huấn luyện Word2Vec. Bạn tải vector đã huấn luyện sẵn.

- **GloVe** — cách tiếp cận phân rã ma trận đồng xuất hiện của Stanford. Có checkpoint 50d, 100d, 200d, 300d. Độ phủ tổng quát tốt. Bài 04 nói riêng về GloVe.
- **fastText** — bản mở rộng Word2Vec của Facebook, embedding cả n-gram ký tự. Xử lý được từ ngoài từ vựng bằng cách ghép các từ con. Bài 04.
- **Word2Vec huấn luyện sẵn trên Google News** — 300d, từ vựng 3M từ, công bố năm 2013. Vẫn được tải về hàng ngày.

### Khi nào Word2Vec vẫn thắng vào năm 2026

- Truy xuất nhẹ theo miền chuyên biệt. Huấn luyện trên tóm tắt y khoa trong một giờ trên laptop, bạn có vector chuyên ngành mà không mô hình tổng quát nào nắm được.
- Đặc trưng kiểu loại suy. `gender_vector = mean(man - woman pairs)`. Trừ nó khỏi các từ khác để có một trục trung tính về giới. Vẫn được dùng trong nghiên cứu công bằng.
- Khả năng diễn giải. 100 chiều đủ nhỏ để vẽ bằng PCA hoặc t-SNE và thực sự thấy các cụm hình thành.
- Bất cứ nơi nào suy luận phải chạy trên thiết bị, không có GPU. Tra cứu Word2Vec chỉ là lấy một hàng.

### Nơi Word2Vec thất bại

Bức tường đa nghĩa. `bank` chỉ có một vector. `river bank` và `financial bank` dùng chung nó. `table` (bảng tính so với đồ nội thất) cũng dùng chung. Một bộ phân loại phía sau không thể phân biệt các nghĩa từ vector đó.

Embedding theo ngữ cảnh (ELMo, BERT, mọi transformer từ đó tới nay) giải quyết việc này bằng cách tạo ra một vector khác nhau cho mỗi lần xuất hiện của từ, dựa trên ngữ cảnh xung quanh. Đó là bước nhảy từ Word2Vec sang BERT: từ tĩnh sang theo ngữ cảnh. Giai đoạn 7 nói về nửa transformer.

Vấn đề ngoài từ vựng là thất bại còn lại. Word2Vec chưa bao giờ thấy `Zoomer-approved` nếu nó không có trong dữ liệu huấn luyện. Không có phương án dự phòng. fastText sửa việc này bằng cách ghép từ con (bài 04).

## Đóng gói

Lưu thành `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Bài tập

1. **Dễ.** Chạy vòng huấn luyện trên một corpus nhỏ (20 câu về mèo và chó). Sau 200 epoch, kiểm tra `nearest(vocab, W, W[vocab["cat"]])` trả về `dog` trong top 3. Nếu không, tăng epoch hoặc mở rộng từ vựng.
2. **Trung bình.** Thêm lấy mẫu con các từ thường gặp. Những từ có tần suất trên `10^-5` bị loại khỏi các cặp huấn luyện với xác suất tỉ lệ với tần suất của chúng. Đo tác động lên độ tương đồng của từ hiếm.
3. **Khó.** Huấn luyện một mô hình trên corpus 20 Newsgroups. Tính hai trục thiên lệch: `he - she` và `doctor - nurse`. Chiếu các từ chỉ nghề nghiệp lên cả hai trục. Báo cáo nghề nào có khoảng cách thiên lệch lớn nhất. Đây là kiểu phép thử mà các nhà nghiên cứu công bằng dùng.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Word embedding | Từ như một vector | Biểu diễn dày, số chiều thấp (thường 100-300) học được từ ngữ cảnh. |
| Skip-gram | Mẹo của Word2Vec | Dự đoán các từ ngữ cảnh từ từ trung tâm. Chậm hơn CBOW, tốt hơn với từ hiếm. |
| Negative sampling | Đường tắt huấn luyện | Thay softmax trên toàn từ vựng bằng phân loại nhị phân với `k` từ ngẫu nhiên. |
| Static embedding | Một vector mỗi từ | Cùng một vector bất kể ngữ cảnh. Thất bại với đa nghĩa. |
| Contextual embedding | Vector nhạy ngữ cảnh | Vector khác nhau cho mỗi lần xuất hiện, dựa trên các từ xung quanh. Thứ mà transformer tạo ra. |
| OOV | Ngoài từ vựng | Từ không thấy trong huấn luyện. Word2Vec không tạo được vector cho những từ này. |

## Đọc thêm

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — bài báo lấy mẫu âm. Ngắn và dễ đọc.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — bản suy dẫn gradient rõ ràng nhất, nếu phần toán của bài báo gốc thấy quá đặc.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — các thiết lập huấn luyện production thực sự hiệu quả.

---

[← Bài trước: Bag of Words, TF-IDF, và Biểu diễn Văn bản](../../02-bag-of-words-tfidf/docs/vi.md) · [Bài sau: GloVe, FastText, và Embedding Subword →](../../04-glove-fasttext-subword/docs/vi.md)
