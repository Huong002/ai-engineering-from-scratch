# Phân tích cảm xúc

> Bài toán NLP kinh điển. Phần lớn những gì bạn cần biết về phân loại văn bản cổ điển đều xuất hiện ở đây.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 02 (BoW + TF-IDF), Giai đoạn 2, Bài 14 (Naive Bayes)
**Time:** ~75 phút

## Vấn đề

"Đồ ăn không ngon lắm." Tích cực hay tiêu cực?

Sentiment nghe có vẻ đơn giản. Một người đánh giá nói họ thích hay không thích điều gì đó. Gán nhãn cho câu. Lý do nó trở thành bài toán NLP kinh điển là vì mọi trường hợp trông dễ đều giấu một trường hợp khó. Phủ định đảo nghĩa. Mỉa mai lật ngược nghĩa. "Not bad at all" là tích cực dù có hai từ mang nghĩa tiêu cực. Emoji mang nhiều tín hiệu hơn cả phần chữ xung quanh. Từ vựng theo lĩnh vực thì quan trọng (`tight` trong review âm nhạc khác `tight` trong review thời trang).

Sentiment analysis (phân tích cảm xúc) là một phòng thí nghiệm thực hành cho NLP cổ điển. Nếu bạn hiểu vì sao mỗi baseline ngây thơ đều có một kiểu thất bại riêng, bạn hiểu vì sao mọi mô hình phong phú hơn ra đời. Bài này tự tay dựng một baseline Naive Bayes, thêm logistic regression, và gọi tên những cái bẫy khiến sentiment trong production thành bài toán chuẩn mực về tuân thủ.

## Khái niệm

Sentiment cổ điển là công thức hai bước.

1. **Biểu diễn.** Chuyển văn bản thành vector đặc trưng. BoW, TF-IDF hoặc n-gram.
2. **Phân loại.** Huấn luyện một mô hình tuyến tính (Naive Bayes, logistic regression, SVM) trên các ví dụ đã có nhãn.

Naive Bayes là mô hình ngớ ngẩn nhất mà vẫn hiệu quả. Giả định mọi đặc trưng độc lập khi đã biết nhãn. Ước lượng `P(word | positive)` và `P(word | negative)` từ số đếm. Lúc suy luận, nhân các xác suất lại. Giả định độc lập "ngây thơ" ấy sai đến mức buồn cười, vậy mà kết quả mạnh đến kinh ngạc. Lý do: với đặc trưng văn bản thưa và labeled data (dữ liệu gán nhãn) vừa phải, classifier (bộ phân loại) quan tâm mỗi từ nghiêng về phía nào hơn là nghiêng bao nhiêu.

Logistic regression khắc phục giả định độc lập. Nó học một trọng số cho mỗi đặc trưng, kể cả trọng số âm. `not good` như một đặc trưng bigram sẽ nhận trọng số âm. Naive Bayes không làm được điều đó với những bigram nó chưa từng thấy nhãn.

```figure
sentiment-logits
```

## Tự tay làm

### Bước 1: một mini-dataset thật

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Nhỏ là có chủ ý. Công việc thực tế dùng hàng chục nghìn ví dụ (IMDb, SST-2, Yelp polarity). Toán học giống hệt nhau.

### Bước 2: multinomial Naive Bayes viết từ đầu

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing (alpha=1.0) là Laplace smoothing. Không có nó, một từ chưa từng xuất hiện trong một lớp sẽ có xác suất bằng không và phép log sẽ nổ tung. `alpha=0.01` phổ biến trong thực tế. `alpha=1.0` là mặc định khi giảng dạy.

### Bước 3: logistic regression viết từ đầu

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization quan trọng ở đây. Đặc trưng văn bản thưa; không có L2 mô hình sẽ học thuộc các ví dụ huấn luyện. Bắt đầu từ `0.01` rồi tinh chỉnh.

### Bước 4: xử lý phủ định (negation handling) — kiểu thất bại

Xét "not good" và "not bad". Một bộ phân loại BoW thấy `{not, good}` và `{not, bad}` rồi học theo cái nào xuất hiện nhiều hơn trong tập huấn luyện. Một bộ phân loại bigram thấy `not_good` và `not_bad` và học chúng như hai đặc trưng riêng biệt. Thường thế là đủ.

Một cách sửa thô hơn nhưng vẫn hiệu quả khi bạn không có bigram: **negation scoping**. Gắn tiền tố `NOT_` cho các token đứng sau một từ phủ định, cho tới dấu câu tiếp theo.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Giờ `good` và `NOT_good` là hai đặc trưng khác nhau. Bộ phân loại có thể cho chúng trọng số ngược nhau. Ba dòng tiền xử lý, một bước nhảy độ chính xác đo được trên các benchmark sentiment.

### Bước 5: những metric đánh giá thật sự quan trọng

Chỉ riêng accuracy là dễ gây hiểu lầm nếu các lớp mất cân bằng. Các corpus sentiment thực tế thường 70-80% tích cực hoặc 70-80% tiêu cực; một bộ phân loại luôn đoán lớp đa số đạt 80% accuracy và vô dụng. Hãy báo cáo từng mục sau:

- **Precision và recall theo từng lớp.** Mỗi lớp một cặp. Lấy trung bình macro để có một con số tôn trọng độ cân bằng lớp.
- **Macro-F1 (metric chính cho dữ liệu mất cân bằng).** Trung bình các điểm F1 theo lớp, trọng số bằng nhau. Dùng cái này thay cho accuracy khi các lớp mất cân bằng.
- **Weighted-F1 (phương án thay thế).** Giống macro nhưng có trọng số theo tần suất lớp. Báo cáo cạnh macro-F1 khi chính sự mất cân bằng mang ý nghĩa kinh doanh.
- **Ma trận nhầm lẫn (confusion matrix).** Số đếm thô. Luôn xem nó trước khi tin bất kỳ metric vô hướng nào; nó cho thấy mô hình nhầm lẫn cặp lớp nào.
- **Mẫu lỗi theo từng lớp.** Lấy 5 dự đoán sai mỗi lớp. Đọc chúng. Không gì thay thế được việc đọc chính các lỗi.

Với dữ liệu mất cân bằng nghiêm trọng (tỉ lệ > 95-5), hãy báo cáo **AUROC** và **AUPRC** thay cho accuracy. AUPRC nhạy hơn với lớp thiểu số, thứ bạn thường quan tâm (spam, gian lận, sentiment hiếm).

**Lỗi phổ biến cần tránh.** Báo cáo micro-F1 thay vì macro-F1 trên dữ liệu mất cân bằng sẽ cho ra một con số trông cao vì nó bị lớp đa số chi phối. Macro-F1 buộc bạn phải thấy hiệu năng của lớp thiểu số.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Áp dụng

scikit-learn làm việc này trong sáu dòng, và làm đúng.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Ba điều cần để ý. `stop_words=None` giữ lại các từ phủ định. `ngram_range=(1, 2)` thêm bigram để `not_good` trở thành một đặc trưng. `sublinear_tf=True` làm dịu các từ bị lặp. Ba cờ này chính là khác biệt giữa baseline chính xác 75% và baseline chính xác 85% trên SST-2.

### Khi nào cần đến transformer

- Phát hiện mỉa mai. Mô hình cổ điển thất bại ở đây. Chấm hết.
- Review dài, nơi cảm xúc đổi giữa văn bản.
- Sentiment theo khía cạnh. "Camera was great but battery was terrible." Bạn cần gán cảm xúc cho từng khía cạnh. Chỉ transformer hoặc mô hình sinh có cấu trúc mới làm được.
- Ngôn ngữ không phải tiếng Anh, ngôn ngữ ít tài nguyên. Multilingual BERT cho bạn một baseline zero-shot miễn phí.

Nếu bạn cần bất kỳ điều nào ở trên, hãy nhảy tới phase 7 (đi sâu transformer). Nếu không, Naive Bayes hoặc logistic regression trên TF-IDF, cộng bigram, cộng xử lý phủ định, là baseline production 2026 của bạn.

### Cái bẫy tái lập (lần nữa)

Huấn luyện lại mô hình sentiment là việc thường ngày. Đánh giá lại chúng thì không. Các con số accuracy công bố trong paper dùng những cách chia dữ liệu cụ thể, tiền xử lý cụ thể, tokenizer cụ thể. Nếu bạn so mô hình mới với một baseline mà không dùng đúng pipeline y hệt, bạn sẽ nhận được chênh lệch gây hiểu lầm. Luôn tạo lại baseline trên pipeline của bạn, đừng lấy con số trong paper.

## Đóng gói

Lưu thành `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Bài tập

1. **Dễ.** Thêm `apply_negation` như một bước tiền xử lý trong pipeline scikit-learn và đo mức thay đổi F1 trên một tập sentiment nhỏ.
2. **Trung bình.** Cài logistic regression có trọng số lớp (truyền `class_weight="balanced"` cho scikit-learn, hoặc tự đạo hàm gradient). Đo tác động trên một tình huống mất cân bằng lớp 90-10 tổng hợp.
3. **Khó.** Xây bộ phát hiện mỉa mai bằng cách huấn luyện một bộ phân loại thứ hai trên phần dư của mô hình sentiment. Ghi lại thiết lập thử nghiệm của bạn. Cảnh báo người đọc khi accuracy của bạn dưới mức ngẫu nhiên (mức ngẫu nhiên cho bài mỉa mai 2 lớp là ~50%, và hầu hết lần thử đầu đều dừng ở đó).

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Polarity | Tích cực hay tiêu cực | Nhãn nhị phân; đôi khi mở rộng sang trung tính hoặc mịn hơn (5 sao). |
| Aspect-based sentiment | Cực tính theo từng khía cạnh | Gán cảm xúc cho các thực thể hoặc thuộc tính cụ thể được nhắc trong văn bản. |
| Negation scoping | Đảo nghĩa các token gần đó | Gắn tiền tố `NOT_` cho các token sau "not" cho tới dấu câu. |
| Laplace smoothing | Cộng 1 vào số đếm | Ngăn các đặc trưng có xác suất bằng không trong Naive Bayes. |
| L2 regularization | Thu nhỏ trọng số | Thêm `lambda * sum(w^2)` vào hàm mất mát. Thiết yếu với đặc trưng văn bản thưa. |

## Đọc thêm

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — bài khảo sát nền tảng. Dài, nhưng bốn phần đầu bao quát toàn bộ phần cổ điển.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — bài báo cho thấy bigram + Naive Bayes rất khó bị đánh bại trên văn bản ngắn.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — tài liệu tham chiếu cho `CountVectorizer`, `TfidfVectorizer` và mọi núm vặn bạn sẽ tinh chỉnh.

---

[← Bài trước: GloVe, FastText, và Embedding Subword](../../04-glove-fasttext-subword/docs/vi.md) · [Bài sau: Named Entity Recognition (nhận dạng thực thể có tên) →](../../06-named-entity-recognition/docs/vi.md)
