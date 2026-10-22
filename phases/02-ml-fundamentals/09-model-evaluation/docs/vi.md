# Đánh giá mô hình

> Một mô hình chỉ tốt bằng đúng cách bạn đo lường nó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 1 (Xác suất và Phân phối, Thống kê cho Học máy), Giai đoạn 2, Bài 01-08
**Time:** ~90 phút

## Mục tiêu học tập

- Triển khai kiểm định chéo (cross-validation) K-fold và stratified K-fold từ đầu và giải thích vì sao chia tầng (stratified split) lại quan trọng với dữ liệu mất cân bằng
- Tính precision, recall, F1, AUC-ROC và các độ đo hồi quy (MSE, RMSE, MAE, R-squared) từ đầu
- Diễn giải đường cong học (learning curve) để chẩn đoán mô hình đang mắc độ chệch cao hay phương sai cao
- Nhận diện những lỗi đánh giá phổ biến, gồm rò rỉ dữ liệu (data leakage), chọn sai độ đo và nhiễm bẩn tập test

## Vấn đề

Bạn đã huấn luyện một mô hình. Nó đạt độ chính xác 95% trên dữ liệu của bạn. Liệu nó có tốt?

Có thể. Có thể không. Nếu 95% dữ liệu của bạn thuộc về một lớp, một mô hình luôn dự đoán lớp đó sẽ đạt độ chính xác 95% trong khi hoàn toàn vô dụng. Nếu bạn đánh giá trên chính dữ liệu đã dùng để huấn luyện, con số 95% là vô nghĩa vì mô hình chỉ đơn thuần học thuộc đáp án. Nếu bộ dữ liệu của bạn có thành phần thời gian mà bạn lại xáo trộn ngẫu nhiên trước khi chia, mô hình có thể đang dùng dữ liệu tương lai để dự đoán quá khứ.

Đánh giá mô hình (model evaluation) là nơi hầu hết các dự án ML đi sai đường. Sai độ đo khiến một mô hình tệ trông có vẻ tốt. Chia sai cho phép mô hình gian lận. So sánh sai khiến bạn chọn phải mô hình tệ hơn. Làm đúng việc đánh giá không phải là tùy chọn. Nó là khác biệt giữa một mô hình chạy được trên production và một mô hình sụp đổ ngay khi gặp dữ liệu thật.

## Khái niệm

### Tập train, validation, test

```mermaid
flowchart LR
    A[Bộ dữ liệu đầy đủ] --> B[Tập train 60-70%]
    A --> C[Tập validation 15-20%]
    A --> D[Tập test 15-20%]
    B --> E[Huấn luyện mô hình]
    E --> C
    C --> F[Tinh chỉnh siêu tham số]
    F --> E
    F --> G[Mô hình cuối cùng]
    G --> D
    D --> H[Báo cáo hiệu năng]
```

Ba phần chia, ba mục đích:

- **Tập huấn luyện**: mô hình học từ dữ liệu này. Nó thấy những ví dụ này trong lúc huấn luyện.
- **Tập validation**: dùng để tinh chỉnh siêu tham số và chọn giữa các mô hình. Mô hình không bao giờ huấn luyện trên dữ liệu này, nhưng các quyết định của bạn lại bị nó ảnh hưởng.
- **Tập test**: chỉ được chạm đúng một lần, ở phút cuối, để báo cáo hiệu năng cuối cùng. Nếu bạn xem hiệu năng test rồi quay lại thay đổi mô hình, nó không còn là tập test nữa. Nó đã trở thành một tập validation thứ hai.

Tập test chính là bảo đảm tập giữ lại (holdout set) của bạn rằng hiệu năng được báo cáo phản ánh đúng cách mô hình sẽ hoạt động trên dữ liệu thực sự chưa từng thấy.

### Kiểm định chéo K-Fold

Với bộ dữ liệu nhỏ, một lần chia train/validation duy nhất vừa lãng phí dữ liệu vừa cho ước lượng nhiễu. Kiểm định chéo K-fold dùng toàn bộ dữ liệu cho cả huấn luyện lẫn validation:

```mermaid
flowchart TB
    subgraph Fold1["Phần 1"]
        direction LR
        V1["Kiểm định"] --- T1a["Huấn luyện"] --- T1b["Huấn luyện"] --- T1c["Huấn luyện"] --- T1d["Huấn luyện"]
    end
    subgraph Fold2["Phần 2"]
        direction LR
        T2a["Huấn luyện"] --- V2["Kiểm định"] --- T2b["Huấn luyện"] --- T2c["Huấn luyện"] --- T2d["Huấn luyện"]
    end
    subgraph Fold3["Phần 3"]
        direction LR
        T3a["Huấn luyện"] --- T3b["Huấn luyện"] --- V3["Kiểm định"] --- T3c["Huấn luyện"] --- T3d["Huấn luyện"]
    end
    subgraph Fold4["Phần 4"]
        direction LR
        T4a["Huấn luyện"] --- T4b["Huấn luyện"] --- T4c["Huấn luyện"] --- V4["Kiểm định"] --- T4d["Huấn luyện"]
    end
    subgraph Fold5["Phần 5"]
        direction LR
        T5a["Huấn luyện"] --- T5b["Huấn luyện"] --- T5c["Huấn luyện"] --- T5d["Huấn luyện"] --- V5["Kiểm định"]
    end
    Fold1 --> R["Điểm trung bình"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R
```

1. Chia dữ liệu thành K fold có kích thước bằng nhau
2. Với mỗi fold, huấn luyện trên K-1 fold và validation trên fold còn lại
3. Lấy trung bình K điểm validation

K=5 hoặc K=10 là những lựa chọn tiêu chuẩn. Mỗi điểm dữ liệu được dùng để validation đúng một lần. Điểm trung bình là một ước lượng ổn định hơn bất kỳ lần chia đơn lẻ nào.

**Stratified K-fold**: giữ nguyên phân bố lớp trong mỗi fold. Nếu bộ dữ liệu của bạn gồm 70% lớp A và 30% lớp B, mỗi fold sẽ có tỷ lệ xấp xỉ như vậy. Điều này quan trọng với dữ liệu mất cân bằng, nơi một lần chia ngẫu nhiên có thể dồn toàn bộ mẫu thiểu số vào một fold.

### Các độ đo phân loại

**Ma trận nhầm lẫn**: nền tảng của mọi thứ. Với phân loại nhị phân:

|  | Dự đoán Positive | Dự đoán Negative |
|--|---|---|
| Thực tế Positive | True Positive (TP) | False Negative (FN) |
| Thực tế Negative | False Positive (FP) | True Negative (TN) |

Từ ma trận này, mọi độ đo khác đều suy ra được:

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN). Tỷ lệ dự đoán đúng. Gây hiểu sai khi các lớp mất cân bằng.
- **Precision** = TP / (TP + FP). Trong tất cả những gì được dự đoán là positive, bao nhiêu thực sự là positive? Dùng khi false positive đắt (ví dụ: bộ lọc spam gắn nhãn spam cho email thật).
- **Recall** (độ nhạy) = TP / (TP + FN). Trong tất cả positive thực tế, ta bắt được bao nhiêu? Dùng khi false negative đắt (ví dụ: tầm soát ung thư bỏ sót khối u).
- **F1 score** = 2 * precision * recall / (precision + recall). Trung bình điều hòa của precision và recall. Cân bằng cả hai khi không cái nào rõ ràng trội hơn.
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve. Vẽ true positive rate so với false positive rate ở nhiều ngưỡng phân loại khác nhau. AUC = 0.5 nghĩa là đoán ngẫu nhiên, AUC = 1.0 nghĩa là tách hoàn hảo. Không phụ thuộc ngưỡng: nó đo mức độ mô hình xếp hạng positive cao hơn negative, bất kể bạn chọn ngưỡng nào.

### Các độ đo hồi quy

- **MSE** (Mean Squared Error) = mean((y_true - y_pred)^2). Phạt sai số lớn theo bậc hai. Nhạy với ngoại lai.
- **RMSE** (Root Mean Squared Error) = sqrt(MSE). Cùng đơn vị với biến mục tiêu. Dễ diễn giải hơn MSE.
- **MAE** (Mean Absolute Error) = mean(|y_true - y_pred|). Xử lý mọi sai số theo tuyến tính. Bền vững hơn MSE trước ngoại lai.
- **R-squared** = 1 - SS_res / SS_tot, với SS_res = sum((y_true - y_pred)^2) và SS_tot = sum((y_true - y_mean)^2). Tỷ lệ phương sai được mô hình giải thích. R^2 = 1.0 là hoàn hảo. R^2 = 0.0 nghĩa là mô hình không tốt hơn việc luôn dự đoán giá trị trung bình. R^2 có thể âm nếu mô hình tệ hơn cả giá trị trung bình.

### Đường cong học

Vẽ điểm huấn luyện và điểm validation như một hàm của kích thước tập huấn luyện:

- **Độ chệch cao (dưới khớp)**: cả hai đường cùng hội tụ về một điểm thấp. Thêm dữ liệu sẽ không giúp gì. Bạn cần một mô hình phức tạp hơn.
- **Phương sai cao (quá khớp)**: điểm huấn luyện cao nhưng điểm validation thấp hơn nhiều. Khoảng cách giữa chúng lớn. Thêm dữ liệu nên giúp ích.

### Đường cong validation

Vẽ điểm huấn luyện và điểm validation như một hàm của một siêu tham số:

- Ở độ phức tạp thấp: cả hai điểm đều thấp (dưới khớp)
- Ở độ phức tạp phù hợp: cả hai điểm đều cao và gần nhau
- Ở độ phức tạp cao: điểm huấn luyện vẫn cao nhưng điểm validation tụt xuống (quá khớp)

Giá trị siêu tham số tối ưu là nơi điểm validation đạt đỉnh.

### Những lỗi đánh giá phổ biến

**Rò rỉ dữ liệu**: thông tin từ tập test lọt vào quá trình huấn luyện. Ví dụ: fit scaler trên toàn bộ dữ liệu trước khi chia, đưa dữ liệu tương lai vào dự đoán chuỗi thời gian, dùng một đặc trưng được suy ra từ biến mục tiêu. Luôn chia trước, rồi mới tiền xử lý.

**Mất cân bằng lớp**: 99% giao dịch là hợp lệ, 1% là gian lận. Một mô hình luôn dự đoán "hợp lệ" đạt độ chính xác 99%. Hãy dùng precision, recall, F1 hoặc AUC-ROC thay thế.

**Sai độ đo**: tối ưu accuracy khi đáng ra phải tối ưu recall (chẩn đoán y khoa), hoặc tối ưu RMSE khi dữ liệu có nhiều ngoại lai (hãy dùng MAE).

**Không dùng chia tầng**: với dữ liệu mất cân bằng, một lần chia ngẫu nhiên có thể đặt rất ít mẫu thiểu số vào fold validation, cho ước lượng không ổn định.

**Test quá thường xuyên**: mỗi lần bạn xem hiệu năng test rồi điều chỉnh, bạn đang quá khớp với tập test. Tập test chỉ dùng một lần.

```figure
precision-recall-threshold
```

## Tự tay làm

### Bước 1: Chia train/validation/test

```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### Bước 2: Kiểm định chéo K-fold và stratified K-fold

```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### Bước 3: Ma trận nhầm lẫn và các độ đo phân loại

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### Bước 4: Các độ đo hồi quy

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### Bước 5: Đường cong học

```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### Bước 6: Một bộ phân loại đơn giản để thử nghiệm, cùng toàn bộ demo

```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== Train/Validation/Test Split ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class distribution: {sum(y_train)}/{len(y_train)} positive")
    print(f"  Val class distribution: {sum(y_val)}/{len(y_val)} positive")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== Classification Metrics ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K-Fold Cross-Validation (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== Stratified K-Fold Cross-Validation (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  Fold scores: {[round(s, 4) for s in strat_scores]}")
    print(f"  Mean: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== Imbalanced Data: Why Accuracy Lies ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  Class distribution: {positives} positive, {len(y_imb) - positives} negative ({positives/len(y_imb)*100:.1f}% positive)")

    always_negative = [0] * len(y_imb)
    print(f"  Always-negative baseline:")
    print(f"    Accuracy:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    Precision: {precision(y_imb, always_negative):.4f}")
    print(f"    Recall:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1 Score:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  Trained model on imbalanced data:")
    print(f"    Accuracy:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    Precision: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    Recall:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1 Score:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== Regression Metrics ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  Mean baseline:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== Learning Curve ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'Size':>6} {'Train':>8} {'Val':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== Statistical Model Comparison ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  Model A (100 epochs) mean: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  Model B (500 epochs) mean: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Paired t-statistic: {t_stat:.4f}")
    print(f"  (|t| > 2.78 for significance at p<0.05 with df=4)")
```

## Áp dụng

Với scikit-learn, việc đánh giá đã nằm sẵn trong quy trình làm việc:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

Bản tự viết từ đầu cho bạn thấy chính xác cross-validation làm gì (không có phép màu, chỉ là vòng lặp for và theo dõi chỉ số), mỗi độ đo được tính ra sao (chỉ là đếm TP/FP/TN/FN), và vì sao stratification lại quan trọng (giữ nguyên tỷ lệ lớp trong từng fold). Bản dùng thư viện thì thêm khả năng chạy song song, nhiều lựa chọn scoring hơn, và tích hợp với pipeline.

## Đóng gói

Bài học này tạo ra:
- `outputs/skill-evaluation.md` - một skill bao quát chiến lược đánh giá cho model phân loại và hồi quy

## Bài tập

1. Cài đặt đường cong precision-recall: vẽ precision theo recall tại các ngưỡng khác nhau. Tính average precision (diện tích dưới đường cong PR). So sánh đường cong PR với đường cong ROC trên một tập dữ liệu mất cân bằng và giải thích khi nào mỗi loại mang nhiều thông tin hơn.
2. Xây dựng một vòng kiểm định chéo lồng (nested cross-validation): vòng ngoài đánh giá hiệu năng model, vòng trong tinh chỉnh siêu tham số. Dùng nó để so sánh hai model một cách công bằng mà không để rò rỉ dữ liệu validation vào bước đánh giá.
3. Cài đặt permutation test để so sánh model: xáo trộn nhãn, huấn luyện lại và đo hiệu năng. Lặp 100 lần để dựng phân phối null. Tính p-value cho hiệu năng quan sát được của model so với phân phối này.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Overfitting | "Học thuộc lòng dữ liệu huấn luyện" | Model bắt lấy cả nhiễu trong dữ liệu huấn luyện, đạt kết quả tốt trên tập huấn luyện nhưng kém trên dữ liệu chưa thấy |
| Cross-validation | "Kiểm tra trên các tập con khác nhau" | Luân phiên một cách có hệ thống phần dữ liệu nào được dùng để validation, rồi lấy trung bình kết quả qua tất cả các lần luân phiên |
| Precision | "Bao nhiêu dự đoán dương là đúng" | TP / (TP + FP): tỷ lệ các dự đoán dương thực sự là dương |
| Recall | "Bao nhiêu mẫu dương thực tế đã tìm ra" | TP / (TP + FN): tỷ lệ các mẫu dương thực tế được nhận diện đúng |
| AUC-ROC | "Model tách các lớp tốt đến đâu" | Diện tích dưới đường cong của true positive rate so với false positive rate trên mọi ngưỡng, từ 0.5 (ngẫu nhiên) đến 1.0 (hoàn hảo) |
| R-squared | "Bao nhiêu phương sai được giải thích" | 1 - (tổng bình phương phần dư / tổng bình phương toàn phần): tỷ lệ phương sai của biến mục tiêu mà model nắm bắt được |
| Data leakage | "Model đã gian lận" | Dùng thông tin trong lúc huấn luyện mà ở thời điểm dự đoán sẽ không có, khiến kết quả đánh giá bị lạc quan |
| Learning curve | "Hiệu năng thay đổi thế nào khi có thêm dữ liệu" | Đồ thị điểm huấn luyện và điểm validation theo kích thước tập huấn luyện, cho thấy model đang underfitting hay overfitting |
| Stratified split | "Giữ tỷ lệ lớp cân bằng" | Chia dữ liệu sao cho mỗi tập con có cùng tỷ lệ mỗi lớp như toàn bộ tập dữ liệu |

## Đọc thêm

- [scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html) - tài liệu tham khảo toàn diện về cross-validation, các độ đo và tinh chỉnh siêu tham số
- [Beyond Accuracy: Precision and Recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) - giải thích rõ ràng kèm ví dụ tương tác
- [A Survey of Cross-Validation Procedures (Arlot & Celisse, 2010)](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) - phân tích chặt chẽ về khi nào và vì sao từng chiến lược CV lại hiệu quả

---

[← Bài trước: Kỹ thuật Đặc trưng & Lựa chọn Đặc trưng](../../08-feature-engineering/docs/vi.md) · [Bài sau: Đánh đổi Bias-Variance →](../../10-bias-variance/docs/vi.md)
