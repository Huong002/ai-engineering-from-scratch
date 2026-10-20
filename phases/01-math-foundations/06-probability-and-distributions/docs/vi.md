# Xác suất và Phân phối

> Probability (xác suất) là ngôn ngữ AI dùng để diễn đạt sự bất định.

**Type:** Learn
**Language:** Python
**Prerequisites:** Giai đoạn 1, Bài 01-04
**Time:** ~75 phút

## Mục tiêu học tập

- Tự cài đặt PMF và PDF từ đầu cho các phân phối Bernoulli, categorical, Poisson, uniform và normal distribution (phân phối chuẩn)
- Tính giá trị kỳ vọng, variance (phương sai), và dùng Định lý Giới hạn Trung tâm để giải thích vì sao phân phối Gauss chiếm ưu thế
- Xây dựng hàm softmax và log-softmax với mẹo ổn định số học (trừ đi max logit)
- Tính cross-entropy loss từ logits và liên hệ nó với negative log-likelihood (likelihood: hợp lý)

## Vấn đề

Một bộ phân loại xuất ra `[0.03, 0.91, 0.06]`. Một language model chọn từ tiếp theo trong 50.000 ứng viên. Một diffusion model tạo ảnh bằng cách lấy mẫu từ các phân phối đã học. Tất cả những điều này chính là xác suất đang vận hành.

Mỗi dự đoán mà model đưa ra đều là một distribution (phân phối) xác suất. Mỗi hàm loss đo xem phân phối dự đoán cách xa phân phối thật bao nhiêu. Mỗi bước huấn luyện điều chỉnh tham số để làm cho một phân phối trông giống phân phối kia hơn. Không có xác suất, bạn không thể đọc một paper ML nào, gỡ lỗi một model nào, hay hiểu vì sao training loss của bạn là NaN.

## Khái niệm

### Biến cố, không gian mẫu (sample space) và xác suất

Không gian mẫu S là tập hợp tất cả các kết quả có thể xảy ra. Một biến cố là một tập con của không gian mẫu. Xác suất ánh xạ các biến cố sang các số từ 0 đến 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Ba tiên đề định nghĩa toàn bộ xác suất:
1. P(A) >= 0 với mọi biến cố A
2. P(S) = 1 (luôn có điều gì đó xảy ra)
3. P(A or B) = P(A) + P(B) khi A và B không thể cùng xảy ra

Mọi thứ còn lại (định lý Bayes, expectation (kỳ vọng), các phân phối) đều suy ra từ ba quy tắc này.

### Xác suất có điều kiện và tính độc lập

P(A|B) là xác suất của A khi biết B đã xảy ra.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Hai biến cố độc lập (independence) khi biết một biến cố không cho bạn biết gì về biến cố kia:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Tung đồng xu là độc lập. Rút bài không hoàn lại thì không.

### Hàm khối xác suất so với hàm mật độ xác suất

Các random variable (biến ngẫu nhiên) rời rạc có một hàm khối xác suất (PMF). Mỗi kết quả có một xác suất cụ thể mà bạn có thể đọc trực tiếp.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Các biến ngẫu nhiên liên tục có một hàm mật độ xác suất (PDF). Mật độ tại một điểm không phải là xác suất. Xác suất đến từ việc tích phân mật độ trên một khoảng.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Sự phân biệt này quan trọng trong ML. Đầu ra của phân loại là PMF (lựa chọn rời rạc). Không gian latent của VAE dùng PDF (liên tục).

### Các phân phối thường gặp

**Bernoulli:** một lần thử, hai kết quả. Mô hình hóa phân loại nhị phân.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** một lần thử, k kết quả. Mô hình hóa phân loại đa lớp (đầu ra softmax).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** mọi kết quả đều có khả năng như nhau. Dùng cho khởi tạo ngẫu nhiên.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** đường cong chuông. Được tham số hóa bởi mean (mu) và variance (phương sai) sigma^2.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** đếm các biến cố hiếm trong một khoảng cố định. Mô hình hóa tốc độ biến cố.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Giá trị kỳ vọng và phương sai

Giá trị kỳ vọng là kết quả trung bình có trọng số.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Phương sai đo độ phân tán quanh giá trị trung bình.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

Trong ML, giá trị kỳ vọng xuất hiện dưới dạng hàm loss (loss trung bình trên phân phối dữ liệu). Phương sai cho bạn biết về độ ổn định của model. Phương sai cao trong gradient nghĩa là huấn luyện bị nhiễu.

### Phân phối đồng thời và phân phối biên

Một phân phối đồng thời P(X, Y) mô tả hai biến ngẫu nhiên cùng lúc.

Ví dụ PMF đồng thời (X = thời tiết, Y = ô):

| | Y=0 (không ô) | Y=1 (ô) | Biên P(X) |
|---|---|---|---|
| X=0 (nắng) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (mưa) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Biên P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Phân phối biên lấy tổng để triệt tiêu biến còn lại:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Tổng theo hàng và cột trong bảng trên chính là các phân phối biên.

### Vì sao phân phối chuẩn xuất hiện khắp mọi nơi

Định lý giới hạn trung tâm: tổng (hoặc trung bình) của nhiều biến ngẫu nhiên độc lập hội tụ về một phân phối chuẩn, bất kể phân phối ban đầu là gì.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Đây là lý do vì sao:
- Sai số đo đạc xấp xỉ chuẩn (nhiều nguồn nhỏ độc lập)
- Khởi tạo trọng số trong mạng neural dùng phân phối chuẩn
- Nhiễu gradient trong SGD xấp xỉ chuẩn (tổng của nhiều gradient mẫu)
- Phân phối chuẩn là phân phối entropy cực đại với mean và variance cho trước

### Log xác suất

Xác suất thô gây ra các vấn đề số học. Nhân nhiều xác suất nhỏ với nhau sẽ nhanh chóng underflow về 0.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log xác suất khắc phục điều này. Phép nhân trở thành phép cộng.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Quy tắc:
- log(a * b) = log(a) + log(b)
- log xác suất luôn <= 0 (vì 0 < P <= 1)
- Càng âm nghĩa là càng ít khả năng xảy ra
- Cross-entropy loss là log xác suất âm của lớp đúng

### Softmax như một phân phối xác suất

Mạng nơ-ron xuất ra các điểm thô (logits). Softmax biến chúng thành một phân phối xác suất hợp lệ.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Mẹo softmax: trừ đi logit lớn nhất trước khi lấy exp để tránh tràn số.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax kết hợp softmax và log để đảm bảo ổn định số học. PyTorch dùng nó bên trong cho hàm mất mát cross-entropy (entropy chéo).

### Lấy mẫu

Lấy mẫu (sampling) nghĩa là rút ra các giá trị ngẫu nhiên từ một phân phối. Trong ML:
- Dropout lấy mẫu ngẫu nhiên các nơ-ron để đưa về 0
- Tăng cường dữ liệu lấy mẫu các phép biến đổi ngẫu nhiên
- Mô hình ngôn ngữ lấy mẫu token tiếp theo từ phân phối dự đoán được
- Mô hình diffusion lấy mẫu nhiễu rồi khử nhiễu dần dần

Lấy mẫu từ các phân phối tùy ý đòi hỏi những kỹ thuật như lấy mẫu biến đổi nghịch đảo, lấy mẫu loại bỏ, hoặc thủ thuật tái tham số hóa (dùng trong VAE).

```figure
gaussian-pdf
```

## Tự tay làm

### Bước 1: Kiến thức cơ bản về xác suất

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Bước 2: Tự viết PMF và PDF từ đầu

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Bước 3: Giá trị kỳ vọng và phương sai

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Bước 4: Lấy mẫu từ các phân phối

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Bước 5: Softmax và log xác suất

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Bước 6: Minh họa Định lý Giới hạn Trung tâm

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Bước 7: Trực quan hóa

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Bản triển khai đầy đủ cùng toàn bộ phần trực quan hóa nằm trong `code/probability.py`.

## Áp dụng

Với NumPy và SciPy, mọi thứ ở trên chỉ còn là một dòng lệnh:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Bạn đã tự xây tất cả những thứ này từ đầu. Giờ bạn đã biết các lệnh gọi thư viện đang làm gì.

## Bài tập

1. Hãy triển khai lấy mẫu biến đổi nghịch đảo cho phân phối mũ. Kiểm chứng bằng cách lấy mẫu 10.000 giá trị rồi so sánh histogram với PDF thật.

2. Hãy lập bảng phân phối đồng thời cho hai con xúc xắc đã bị làm lệch. Tính các phân phối biên và kiểm tra xem hai con xúc xắc có độc lập không.

3. Hãy tính mất mát cross-entropy (entropy chéo) cho một bộ phân loại 5 lớp xuất ra logits `[2.0, 0.5, -1.0, 3.0, 0.1]` khi lớp đúng là chỉ số 3. Sau đó kiểm chứng kết quả bằng `nn.CrossEntropyLoss` của PyTorch.

4. Hãy viết một hàm nhận danh sách log xác suất và trả về chuỗi có khả năng nhất, tổng log xác suất, và xác suất thô tương đương. Kiểm thử nó với một câu 50 từ, trong đó mỗi từ có xác suất 0.01.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Sample space | "Mọi khả năng" | Tập S gồm mọi kết cục có thể xảy ra của một phép thử |
| PMF | "Hàm xác suất" | Hàm cho biết xác suất chính xác của từng kết cục rời rạc, tổng bằng 1 |
| PDF | "Đường cong xác suất" | Hàm mật độ cho biến liên tục. Lấy tích phân trên một khoảng để được xác suất |
| Conditional probability | "Xác suất khi biết điều gì đó" | P(A\|B) = P(A and B) / P(B). Nền tảng của tư duy Bayes và định lý Bayes |
| Independence | "Chúng không ảnh hưởng lẫn nhau" | P(A and B) = P(A) * P(B). Biết một biến cố chẳng cho bạn thông tin gì về biến cố kia |
| Expected value | "Trung bình cộng" | Tổng có trọng số theo xác suất của mọi kết cục. Hàm mất mát chính là một giá trị kỳ vọng |
| Variance | "Mức phân tán" | Kỳ vọng bình phương độ lệch so với giá trị trung bình. Phương sai cao = ước lượng nhiễu, không ổn định |
| Normal distribution | "Đường cong hình chuông" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Xuất hiện khắp nơi nhờ CLT |
| Central Limit Theorem | "Trung bình thành chuẩn" | Trung bình của nhiều mẫu độc lập hội tụ về phân phối chuẩn, bất kể phân phối nguồn là gì |
| Joint distribution | "Hai biến cùng lúc" | P(X, Y) mô tả xác suất của mọi tổ hợp kết cục của X và Y |
| Marginal distribution | "Cộng dồn biến kia ra ngoài" | P(X) = sum_y P(X, Y). Khôi phục phân phối của một biến từ phân phối đồng thời |
| Log probability | "Log của xác suất" | log P(x). Biến tích thành tổng, tránh tràn số dưới trong các chuỗi dài |
| Softmax | "Biến điểm thành xác suất" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Ánh xạ logits giá trị thực sang một phân phối xác suất hợp lệ |
| Cross-entropy | "Hàm mất mát" | -sum(p_true * log(p_predicted)). Đo mức khác biệt giữa hai phân phối. Càng thấp càng tốt |
| Logits | "Đầu ra thô của mô hình" | Điểm chưa chuẩn hóa trước softmax. Được đặt tên theo hàm logistic |
| Sampling | "Rút giá trị ngẫu nhiên" | Sinh giá trị theo một phân phối xác suất. Cách mô hình tạo ra đầu ra |

## Đọc thêm

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - chứng minh trực quan vì sao các giá trị trung bình trở nên chuẩn
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - tài liệu tham khảo cô đọng bao quát toàn bộ nội dung ở đây và hơn thế nữa
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - vì sao ổn định số học quan trọng và cách đạt được nó
