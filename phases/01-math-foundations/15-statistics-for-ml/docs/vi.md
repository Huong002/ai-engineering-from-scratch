# Thống kê cho Học máy

> Thống kê (statistics) là cách bạn biết liệu model của mình thực sự hoạt động hay chỉ gặp may.

**Type:** Build
**Language:** Python
**Prerequisites:** Giai đoạn 1, Bài 06 (Xác suất và Phân phối), 07 (Định lý Bayes)
**Time:** ~120 phút

## Mục tiêu học tập

- Tính các thống kê mô tả, tương quan Pearson/Spearman và ma trận hiệp phương sai từ đầu
- Thực hiện kiểm định giả thuyết (t-test, chi-squared) và diễn giải p-value cùng khoảng tin cậy cho đúng
- Dùng bootstrap resampling để dựng khoảng tin cậy cho bất kỳ metric nào mà không cần giả định về phân phối
- Phân biệt ý nghĩa thống kê với ý nghĩa thực tiễn bằng các thước đo effect size

## Vấn đề

Bạn huấn luyện hai model. Model A được 0.87 trên tập test. Model B được 0.89. Bạn triển khai Model B. Ba tuần sau, các metric trên production tệ hơn trước. Chuyện gì đã xảy ra?

Model B không thực sự vượt Model A. Chênh lệch 0.02 chỉ là nhiễu. Tập test của bạn quá nhỏ, hoặc phương sai (variance) quá cao, hoặc cả hai. Bạn đã đưa sự ngẫu nhiên vào và khoác cho nó áo của một cải tiến.

Chuyện này xảy ra liên tục. Bảng xếp hạng Kaggle đảo lộn. Các bài báo không tái lập được. Các A/B test tuyên bố người thắng chỉ dựa trên vài trăm mẫu. Nguyên nhân gốc luôn giống nhau: ai đó đã bỏ qua phần thống kê.

Thống kê cho bạn bộ công cụ để phân biệt tín hiệu với nhiễu. Nó cho bạn biết khi nào một khác biệt là thật, bạn nên tin đến mức nào, và cần bao nhiêu dữ liệu trước khi có thể tin một kết quả. Mọi pipeline ML, mọi phép so sánh model, mọi thí nghiệm đều cần thống kê. Không có nó, bạn chỉ đang đoán.

## Khái niệm

### Thống kê mô tả: Tóm tắt dữ liệu của bạn

Trước khi mô hình hóa bất cứ thứ gì, bạn cần biết dữ liệu của mình trông như thế nào. Thống kê mô tả nén một tập dữ liệu thành vài con số nắm bắt hình dạng của nó.

**Các thước đo xu hướng trung tâm** trả lời câu hỏi "điểm giữa ở đâu?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Trung bình (mean) là điểm cân bằng. Trung vị (median) là mốc giữa. Khi hai giá trị này tách xa nhau, phân phối (distribution) của bạn bị lệch. Phân phối thu nhập có mean >> median (lệch phải do các tỷ phú). Phân phối loss khi huấn luyện thường có mean << median (lệch trái do các mẫu dễ).

**Các thước đo độ phân tán** trả lời câu hỏi "dữ liệu phân tán đến mức nào?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Phân vị (quantile)** chia dữ liệu đã sắp xếp thành 100 phần bằng nhau. Phân vị thứ 25 (Q1) nghĩa là 25% giá trị nằm dưới mốc này. Phân vị thứ 50 là trung vị. Phân vị thứ 75 là Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

Trong ML, bạn quan tâm đến phân vị cho độ trễ suy luận, cho phân phối độ tin cậy của dự đoán, và để hiểu phân phối sai số. Một model có sai số trung bình thấp nhưng sai số P99 khủng khiếp có thể vô dụng với các ứng dụng yêu cầu an toàn nghiêm ngặt.

**Thống kê mẫu so với tổng thể.** Khi tính phương sai từ một mẫu, hãy chia cho (n-1) thay vì n. Đây là hiệu chỉnh Bessel. Nó bù cho việc trung bình mẫu của bạn không phải trung bình thật của tổng thể. Với n ở mẫu số, bạn luôn đánh giá thấp phương sai thật. Với (n-1), ước lượng là không lệch.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

Trên thực tế: nếu n lớn (hàng nghìn mẫu), khác biệt không đáng kể. Nếu n nhỏ (vài chục mẫu), nó có ý nghĩa.

### Tương quan: Cách các biến dịch chuyển cùng nhau

Tương quan (correlation) đo độ mạnh và hướng của quan hệ tuyến tính giữa hai biến.

**Hệ số tương quan Pearson** đo liên kết tuyến tính:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson giả định quan hệ là tuyến tính và cả hai biến xấp xỉ phân phối chuẩn. Nó nhạy với giá trị ngoại lai (outlier). Một điểm cực trị duy nhất có thể kéo r từ 0.1 lên 0.9.

**Tương quan hạng Spearman** đo liên kết đơn điệu:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Khi nào dùng loại nào:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Quy tắc vàng:** tương quan không ngụ ý nhân quả. Doanh số kem và số ca chết đuối tương quan vì cả hai đều tăng vào mùa hè. Độ chính xác của model và số tham số tương quan, nhưng thêm tham số không tự động cải thiện độ chính xác (xem: overfitting).

### Ma trận hiệp phương sai

Hiệp phương sai giữa hai biến đo mức độ chúng biến thiên cùng nhau:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Với d đặc trưng, ma trận hiệp phương sai C là ma trận d x d trong đó C[i][j] = Cov(feature_i, feature_j). Các phần tử trên đường chéo C[i][i] là phương sai của từng đặc trưng.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Liên hệ với PCA.** PCA phân rã trị riêng ma trận hiệp phương sai. Các vector riêng là các thành phần chính (hướng có phương sai lớn nhất). Các trị riêng cho bạn biết mỗi thành phần nắm bao nhiêu phương sai. Đây chính là nội dung Bài 10, nhưng giờ bạn thấy vì sao ma trận hiệp phương sai là thứ đúng để phân rã: nó mã hóa mọi quan hệ tuyến tính theo cặp trong dữ liệu của bạn.

**Liên hệ với tương quan.** Ma trận tương quan là ma trận hiệp phương sai của các biến đã chuẩn hóa (mỗi biến chia cho độ lệch chuẩn (standard deviation) của nó). Tương quan chuẩn hóa hiệp phương sai để mọi giá trị rơi vào [-1, 1].

### Kiểm định giả thuyết

Kiểm định giả thuyết là một khuôn khổ để ra quyết định trong điều kiện không chắc chắn. Bạn bắt đầu với một tuyên bố, thu thập dữ liệu, rồi xác định xem dữ liệu có nhất quán với tuyên bố đó không.

**Cách thiết lập:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**P-value** là xác suất quan sát được dữ liệu cực đoan như dữ liệu bạn đã thấy, với giả định H0 đúng. Nó KHÔNG phải xác suất H0 đúng. Đây là hiểu nhầm phổ biến nhất trong thống kê.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Khoảng tin cậy** cho một khoảng giá trị hợp lý cho một tham số:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Độ rộng của khoảng tin cậy cho bạn biết về độ chính xác. Khoảng rộng nghĩa là bất định cao. Khoảng hẹp nghĩa là ước lượng của bạn chính xác (nhưng không nhất thiết đúng, nếu dữ liệu của bạn bị lệch).

### Kiểm định t

Kiểm định t so sánh các trung bình. Có vài biến thể.

**Kiểm định t một mẫu:** trung bình tổng thể có khác một giá trị giả định không?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Kiểm định t hai mẫu (độc lập):** hai trung bình nhóm có khác nhau không?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Kiểm định t ghép cặp:** khi các phép đo đi theo cặp (cùng một model được đánh giá trên cùng các split dữ liệu):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

Trong ML, kiểm định t ghép cặp rất phổ biến: bạn chạy cả hai model trên cùng 10 fold cross-validation và so sánh điểm của chúng theo từng cặp.

### Kiểm định chi-squared

Kiểm định chi-squared kiểm tra xem tần suất quan sát được có khớp với tần suất kỳ vọng không. Hữu ích cho dữ liệu phân loại.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### Kiểm thử A/B cho mô hình ML

Kiểm thử A/B trong ML không giống kiểm thử A/B trên web. Việc so sánh mô hình có những thách thức riêng:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Quy trình:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Ý nghĩa thống kê và ý nghĩa thực tiễn

Một kết quả có thể có ý nghĩa thống kê (significance) nhưng lại vô nghĩa trên thực tế. Khi có đủ dữ liệu, ngay cả một khác biệt nhỏ nhặt cũng trở nên có ý nghĩa thống kê.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Kích thước hiệu ứng** định lượng mức độ lớn của khác biệt, độc lập với cỡ mẫu (sample size):

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Luôn báo cáo cả p-value lẫn kích thước hiệu ứng. p-value cho bạn biết khác biệt có thật hay không. Kích thước hiệu ứng cho bạn biết nó có quan trọng hay không.

### Vấn đề so sánh bội

Khi bạn kiểm định nhiều giả thuyết, một số sẽ "có ý nghĩa" một cách ngẫu nhiên. Nếu bạn kiểm định 20 thứ ở alpha = 0.05, bạn kỳ vọng có 1 dương tính giả ngay cả khi không có gì là thật.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Hiệu chỉnh Bonferroni:** chia alpha cho số phép kiểm định.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

Trong ML, điều này quan trọng khi bạn so sánh một mô hình trên nhiều chỉ số, thử nhiều cấu hình siêu tham số, hoặc đánh giá trên nhiều bộ dữ liệu.

### Phương pháp bootstrap

Bootstrap ước lượng phân phối mẫu của một thống kê bằng cách lấy mẫu lại dữ liệu của bạn có hoàn lại. Không cần giả định gì về phân phối nền.

**Thuật toán:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Khoảng tin cậy bootstrap (phương pháp phân vị):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Vì sao bootstrap quan trọng với ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap để so sánh mô hình:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Cách này vững hơn kiểm định t ghép cặp vì nó không đưa ra giả định nào về phân phối.

### Kiểm định tham số và phi tham số

**Kiểm định tham số** giả định một phân phối cụ thể (thường là phân phối chuẩn):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Kiểm định phi tham số** không đưa ra giả định nào về phân phối:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Khi nào dùng phi tham số:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Khi nào dùng tham số:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

Trong thí nghiệm ML, bạn thường có n nhỏ (5 hoặc 10 fold cross-validation), nên các kiểm định phi tham số như Wilcoxon signed-rank thường phù hợp hơn t-test.

### Định lý giới hạn trung tâm: hệ quả thực tiễn

CLT nói rằng phân phối của trung bình mẫu tiến về phân phối chuẩn khi n tăng, bất kể phân phối của tổng thể.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Vì sao điều này quan trọng với ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Những gì CLT KHÔNG làm:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Những lỗi thống kê thường gặp trong các bài báo ML

1. **Kiểm thử trên tập huấn luyện.** Chắc chắn dẫn đến overfit. Luôn giữ lại dữ liệu mà mô hình không bao giờ thấy trong quá trình huấn luyện.

2. **Không có khoảng tin cậy.** Báo cáo một con số độ chính xác duy nhất mà không kèm bất định khiến kết quả không thể tái lập và không thể kiểm chứng.

3. **Bỏ qua so sánh bội.** Thử 50 cấu hình rồi báo cáo cấu hình tốt nhất mà không hiệu chỉnh sẽ làm phồng tỷ lệ dương tính giả.

4. **Nhầm lẫn giữa ý nghĩa thống kê và ý nghĩa thực tiễn.** p-value 0.001 cho một cải thiện độ chính xác 0.01% là không có ý nghĩa.

5. **Dùng độ chính xác trên dữ liệu mất cân bằng.** Độ chính xác 99% trên bộ dữ liệu có 99% lớp âm nghĩa là mô hình chẳng học được gì. Hãy dùng precision, recall, F1 hoặc AUC.

6. **Chọn chỉ số theo ý mình.** Chỉ báo cáo đúng chỉ số mà mô hình của bạn thắng. Đánh giá trung thực phải báo cáo mọi chỉ số liên quan.

7. **Rò rỉ thông tin giữa các tập train/test.** Chuẩn hóa trước khi chia tập, hoặc dùng dữ liệu tương lai để dự đoán quá khứ.

8. **Tập kiểm thử nhỏ mà không có ước lượng phương sai.** Đánh giá trên 100 mẫu rồi tuyên bố cải thiện 2% là nhiễu, không phải tín hiệu.

9. **Giả định tính độc lập khi dữ liệu không độc lập.** Ảnh y tế từ cùng một bệnh nhân, nhiều câu từ cùng một tài liệu. Các quan sát trong một nhóm có tương quan với nhau.

10. **P-hacking.** Thử các kiểm định, tập con, hoặc tiêu chí loại trừ khác nhau cho đến khi có p < 0.05. Kết quả chỉ là sản phẩm phụ của chính việc tìm kiếm.

## Tự tay làm

Bạn sẽ triển khai:

1. **Thống kê mô tả từ đầu** (mean, median, mode, standard deviation, percentiles, IQR)
2. **Các hàm tương quan** (Pearson và Spearman, cùng ma trận hiệp phương sai)
3. **Kiểm định giả thuyết (hypothesis testing)** (t-test một mẫu, t-test hai mẫu, kiểm định chi bình phương)
4. **Khoảng tin cậy bootstrap** (cho bất kỳ thống kê nào, không cần giả định)
5. **Trình mô phỏng kiểm thử A/B** (sinh dữ liệu, kiểm định, kiểm tra lỗi loại I và loại II)
6. **Demo ý nghĩa thống kê và ý nghĩa thực tiễn** (cho thấy n lớn khiến mọi thứ đều "có ý nghĩa")

Tất cả từ đầu, chỉ dùng `math` và `random`. Không numpy, không scipy.

```figure
f3-bootstrap-resample
```

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa thực sự |
|---|---|
| Mean | Tổng các giá trị chia cho số lượng. Nhạy với giá trị ngoại lai. |
| Median | Giá trị nằm giữa của dữ liệu đã sắp xếp. Vững với ngoại lai. |
| Standard deviation | Căn bậc hai của phương sai. Đo độ trải theo đơn vị gốc. |
| Percentile | Giá trị mà dưới nó có một phần trăm dữ liệu cho trước rơi vào. |
| IQR | Khoảng tứ phân vị. Q3 trừ Q1. Độ trải của 50% dữ liệu ở giữa. |
| Pearson correlation | Đo mức liên hệ tuyến tính giữa hai biến. Miền giá trị [-1, 1]. |
| Spearman correlation | Đo mức liên hệ đơn điệu dựa trên hạng. |
| Covariance matrix | Ma trận các hiệp phương sai từng cặp giữa mọi đặc trưng. |
| Null hypothesis | Giả định mặc định rằng không có hiệu ứng hay không có khác biệt. |
| p-value | Xác suất gặp dữ liệu cực đoan như vậy nếu giả thuyết không là đúng. |
| Confidence interval | Miền các giá trị hợp lý cho một tham số ở một mức tin cậy cho trước. |
| t-test | Kiểm định xem các trung bình có khác nhau đáng kể không. Dùng phân phối t. |
| Chi-squared test | Kiểm định xem tần suất quan sát được có khác tần suất kỳ vọng không. |
| Effect size | Độ lớn của một khác biệt, độc lập với cỡ mẫu. Cohen's d là phổ biến. |
| Bonferroni correction | Chia ngưỡng ý nghĩa cho số phép kiểm định để kiểm soát dương tính giả. |
| Bootstrap | Lấy mẫu lại có hoàn lại để ước lượng phân phối mẫu. |
| Type I error | Bác bỏ H0 khi H0 đúng. |
| Type II error | Không bác bỏ được H0 khi H0 sai. |
| Statistical power | Xác suất bác bỏ đúng một H0 sai. Power = 1 trừ tỷ lệ lỗi loại II. |
| Central limit theorem | Trung bình mẫu hội tụ về phân phối chuẩn khi cỡ mẫu tăng. |
| Parametric test | Giả định một phân phối cụ thể cho dữ liệu (thường là phân phối chuẩn). |
| Non-parametric test | Không đưa ra giả định nào về phân phối. Làm việc trên hạng hoặc dấu. |

---

[← Bài trước: Norms và Khoảng cách](../../14-norms-and-distances/docs/vi.md) · [Bài sau: Các phương pháp lấy mẫu →](../../16-sampling-methods/docs/vi.md)
