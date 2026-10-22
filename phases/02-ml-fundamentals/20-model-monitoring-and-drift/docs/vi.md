# Giám sát mô hình trong production — data drift (trôi lệch dữ liệu) và concept drift (trôi lệch khái niệm)

> Một mô hình đã triển khai sẽ hỏng mà không hề có thông báo lỗi. Nó vẫn trả về điểm số trong khi thế giới dần rời xa dữ liệu huấn luyện của nó, và chỉ có giám sát mới cho thấy khoảng cách đó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 2, Bài 09 (Đánh giá mô hình), 16 (Phát hiện bất thường) và 19 (Thiết kế hệ thống ML)
**Time:** ~110 phút

## Mục tiêu học tập

- Chọn tín hiệu giám sát ở bốn lớp: sức khoẻ hệ thống, đầu vào, dự đoán và kết quả
- Phân biệt covariate drift (dịch chuyển hiệp biến), prior drift (trôi lệch tiên nghiệm) và concept drift theo phần nào của phân phối đồng thời thay đổi
- Tự viết từ đầu Population Stability Index, thống kê Kolmogorov-Smirnov hai mẫu và kiểm định chi-square
- So sánh một cửa sổ tham chiếu cố định với các cửa sổ hiện tại trượt, và giải thích cách kích thước cửa sổ làm thay đổi nhiễu và độ trễ
- Thiết kế một chính sách alerting (cảnh báo) giới hạn báo động giả bằng effect size, hiệu chỉnh đa kiểm định và quy tắc kiên trì
- Quyết định loại trôi lệch nào cần retraining (huấn luyện lại), loại nào cần nhãn mới, và loại nào không cần làm gì

## Vấn đề

Một mô hình rủi ro thanh toán đi vào vận hành với độ chính xác 95 phần trăm. Hai tháng sau, một chiến dịch marketing mang về nhóm người dùng app có giỏ hàng lớn hơn. Bốn tháng sau, kẻ gian đổi chiêu trò, và những giao dịch từng an toàn bắt đầu biến thành chargeback.

Dịch vụ mô hình trả HTTP 200 cho mọi request trong suốt thời gian đó. Độ trễ bình thường, và log không có lỗi. Dấu hiệu rắc rối đầu tiên là một báo cáo tài chính về chargeback, nhiều tuần sau khi thiệt hại đã xảy ra.

Giám sát phần mềm kiểm tra xem dịch vụ có trả lời không. Model monitoring (giám sát mô hình) kiểm tra xem các câu trả lời còn đúng không, và dữ liệu còn giống dữ liệu đã huấn luyện mô hình không. Bài này xây dựng phần thống kê, các cửa sổ và chính sách cảnh báo cho công việc thứ hai đó. Bạn chạy chúng trên một luồng dữ liệu tổng hợp với hai thay đổi được cấy vào và xem bộ giám sát nào phản ứng với thay đổi nào.

## Khái niệm

### Cần giám sát những gì

Hãy giám sát bốn lớp. Ba lớp đầu cho cảnh báo sớm. Chỉ lớp thứ tư đo tính đúng đắn, và nó đến muộn.

| Lớp | Tín hiệu | Có sẵn | Câu hỏi mà nó trả lời |
|---|---|---|---|
| Sức khoẻ hệ thống | độ trễ p50 và p99, tỷ lệ lỗi, thông lượng, bộ nhớ | Ngay lập tức | Dịch vụ có trả lời không? |
| Đầu vào | Phân phối từng đặc trưng, tỷ lệ giá trị thiếu, schema, danh mục mới | Ngay lập tức | Dữ liệu trực tiếp có giống dữ liệu huấn luyện không? |
| Dự đoán | Phân phối điểm số, tỷ lệ dương, phần chia của các câu trả lời mặc định | Ngay lập tức | Mô hình có hành xử như trước không? |
| Kết quả | Độ chính xác, precision, hiệu chuẩn, chỉ số kinh doanh | Sau độ trễ nhãn | Mô hình còn đúng không? |

Giám sát đầu vào là chủ đề chính của bài này, vì nó hoạt động trước khi nhãn đến. Nó có một giới hạn mà phần demo cho thấy: vài thay đổi không đụng gì tới đầu vào.

### Ba loại trôi lệch

Viết phân phối đồng thời của đầu vào `x` và nhãn `y` theo hai cách:

```text
P(x, y) = P(x) * P(y | x) = P(y) * P(x | y)
```

Mỗi loại trôi lệch thay đổi một nhân tố khác nhau:

| Trôi lệch | Thay đổi gì | Giữ nguyên gì | Ví dụ thanh toán | Bộ giám sát nào thấy |
|---|---|---|---|---|
| Covariate drift | `P(x)` | `P(y \| x)` | Nhiều người dùng app hơn với giỏ hàng lớn hơn | Bộ giám sát đầu vào |
| Prior drift | `P(y)` | `P(x \| y)` | Tỷ lệ gian lận tăng gấp đôi trong một đợt tấn công, và gian lận trông vẫn như cũ | Bộ giám sát dự đoán và kết quả |
| Concept drift | `P(y \| x)` | Có thể giữ nguyên | Cùng một giỏ hàng giờ có rủi ro gian lận khác | Chỉ bộ giám sát kết quả |

Gama et al. gọi thay đổi ở `P(y | x)` là real concept drift. Họ gọi thay đổi ở `P(x)` mà để yên `P(y | x)` là virtual drift. Khảo sát của họ cũng phân loại trôi lệch theo cách nó diễn ra theo thời gian: đột ngột, tăng dần, dần dần, hoặc như một khái niệm quay trở lại.

Covariate drift không phải lúc nào cũng gây hại. Nếu mô hình đã học `P(y | x)` tốt, nó vẫn đúng với từng đầu vào. Nó vẫn có thể mất độ chính xác khi lưu lượng dịch chuyển vào những vùng có ít dữ liệu huấn luyện hoặc gần ranh giới quyết định.

### Nhãn đến muộn

Kết quả đến sau dự đoán. Độ trễ phụ thuộc vào bài toán:

| Bài toán | Khi nào nhãn đến |
|---|---|
| Click quảng cáo | Vài giây |
| Churn trong 30 ngày tới | 30 ngày |
| Gian lận thẻ | Khi chargeback đến, thường là nhiều tuần sau |
| Vỡ nợ khoản vay | Nhiều tháng hoặc nhiều năm |

Trong suốt thời gian trễ, bạn có đầu vào và dự đoán nhưng không có kết quả. Hãy giám sát đầu vào và dự đoán, join nhãn khi chúng đến, và tính các chỉ số kết quả chỉ trên phần đã có nhãn. Demo dùng độ trễ nhãn 500 sự kiện, nên bộ giám sát kết quả luôn nhìn về quá khứ 500 sự kiện.

### Cửa sổ tham chiếu và cửa sổ hiện tại

Mỗi kiểm định trôi lệch so sánh hai mẫu:

- **Cửa sổ tham chiếu.** Dữ liệu huấn luyện, hoặc một giai đoạn production mà bạn đã kiểm tra bằng tay. Nó giữ cố định cho tới khi bạn huấn luyện lại.
- **Cửa sổ hiện tại.** Các sự kiện gần nhất, hoặc `N` sự kiện cuối, hoặc `T` giờ cuối.

Một cửa sổ trượt tiến về phía trước với bước nhỏ hơn kích thước của nó, nên các cửa sổ liên tiếp chồng lên nhau. Cửa sổ nhảy (tumbling) tiến đúng bằng kích thước của nó. Kích thước cửa sổ là một đánh đổi. Cửa sổ nhỏ phản ứng nhanh nhưng cho điểm cả phần nhiễu thành trôi lệch. Cửa sổ lớn thì ổn định nhưng báo thay đổi muộn.

Tính mùa vụ phá vỡ các cửa sổ ngây thơ. Lưu lượng thứ Hai khác thứ Bảy ở nhiều sản phẩm. Hãy cho cửa sổ tham chiếu bao trùm một chu kỳ đầy đủ, hoặc so sánh mỗi cửa sổ với cùng giai đoạn ở chu kỳ trước đó.

### Population Stability Index

PSI đến từ chấm điểm tín dụng. Cắt các giá trị tham chiếu thành các bin, thường là mười bin theo phân vị thập phân của tham chiếu, sao cho mỗi bin chứa 10 phần trăm mẫu tham chiếu. Đếm phần chia của các giá trị hiện tại trong mỗi bin. Rồi cộng một số hạng cho mỗi bin:

```text
PSI = sum over bins of (current_share - reference_share) * ln(current_share / reference_share)
```

Một bin có phần chia bằng zero làm log tiến tới vô cùng, nên hãy kẹp mọi phần chia về một giá trị nhỏ như 0.0001. PSI đối xứng. Nó bằng KL divergence từ hiện tại tới tham chiếu cộng KL divergence từ tham chiếu tới hiện tại.

Người thực hành đọc PSI theo các dải cố định: dưới 0.1 là ổn định, 0.1 đến 0.25 là dịch chuyển vừa, trên 0.25 là dịch chuyển đáng kể. Hãy coi những dải này như một quy ước, không phải một kiểm định. PSI tăng theo số bin và giảm theo kích thước mẫu. Trong demo, các cửa sổ 500 sự kiện không có trôi lệch cho điểm từ 0.01 đến 0.04.

### Thống kê Kolmogorov-Smirnov

KS statistic (thống kê KS) ở dạng hai mẫu, ký hiệu `D`, là khoảng cách dọc lớn nhất giữa hai hàm phân phối tích luỹ thực nghiệm:

```text
D = max over x of | F_reference(x) - F_current(x) |
```

`D` bằng 0 với hai mẫu giống hệt và bằng 1 với hai mẫu không chồng lấn. Nó không cần bin. p-value (giá trị p) đến từ phân phối Kolmogorov với kích thước mẫu hiệu dụng `n * m / (n + m)`.

Với mẫu lớn, p-value tiến về zero với một khoảng cách mà chẳng ai quan tâm. Một dịch chuyển 0.02 độ lệch chuẩn là "có ý nghĩa" với một triệu sự kiện mỗi cửa sổ. Hãy dùng chính `D` như một effect size, và chỉ cảnh báo khi cả `D` lẫn p-value đều vượt ngưỡng của chúng.

### Kiểm định chi-square cho đặc trưng phân loại

Với một đặc trưng phân loại, ví dụ kênh thanh toán, hãy lập bảng hai hàng (tham chiếu, hiện tại) và một cột cho mỗi danh mục. Số đếm kỳ vọng ở mỗi ô đến từ tổng hàng và tổng cột:

```text
expected[r][c] = row_total[r] * column_total[c] / grand_total
chi2 = sum over cells of (observed - expected)^2 / expected
```

Kiểm định có `k - 1` bậc tự do với `k` danh mục. Một danh mục chỉ xuất hiện trong cửa sổ hiện tại sẽ thêm một số hạng lớn, nên kiểm định phản ứng với các giá trị mới. Cramér's V là effect size. Với bảng hai hàng, nó bằng căn bậc hai của `chi2 / grand_total`.

### Ngưỡng cảnh báo và báo động giả

Mỗi kiểm định có một tỷ lệ báo động giả `alpha` khi không có gì thay đổi. Giám sát chạy rất nhiều kiểm định. Với 50 đặc trưng được kiểm tra mỗi giờ ở `alpha = 0.05`, bạn kỳ vọng khoảng 60 báo động giả mỗi ngày khi không có gì thay đổi. Người ta bỏ đọc cảnh báo từ lâu trước khi chạm mức đó.

Bốn quy tắc giữ cho tỷ lệ cảnh báo còn hữu ích:

1. **Hiệu chỉnh cho đa kiểm định.** Hiệu chỉnh Bonferroni chia `alpha` cho số kiểm định. Demo dùng `alpha` họ 0.01 trên hai kiểm định, nên mỗi kiểm định dùng 0.005.
2. **Yêu cầu một effect size.** Chỉ cảnh báo khi PSI, `D` hoặc Cramér's V cũng vượt ngưỡng.
3. **Yêu cầu tính kiên trì.** Chỉ cảnh báo sau `k` cửa sổ trôi lệch liên tiếp. Demo dùng `k = 2`.
4. **Tách cảnh cáo khỏi báo động.** Cảnh cáo đi vào dashboard. Báo động đi tới một con người.

Demo cho thấy quy tắc 2 hoạt động. Một cửa sổ sạch có p-value chi-square 0.0009, dưới `alpha` của mỗi kiểm định. Cramér's V của nó dưới 0.1, nên chính sách ghi một cảnh cáo, không phải báo động.

### Kích hoạt huấn luyện lại

Có ba kiểu kích hoạt huấn luyện lại thường gặp:

- **Theo lịch.** Huấn luyện lại mỗi tuần hoặc mỗi tháng. Cách này đơn giản, nhưng nó huấn luyện lại khi không có gì thay đổi và chờ khi có thay đổi.
- **Theo trôi lệch.** Huấn luyện lại khi trôi lệch đầu vào trên các đặc trưng quan trọng vẫn ở trên ngưỡng.
- **Theo hiệu năng.** Huấn luyện lại khi một chỉ số kết quả xuống dưới giá trị tham chiếu trừ đi một dung sai.

Loại trôi lệch quyết định việc huấn luyện lại có thể làm được gì:

| Trôi lệch | Hành động hữu ích |
|---|---|
| Covariate drift, độ chính xác ổn định | Thường không cần gì. Tiếp tục theo dõi chỉ số kết quả. |
| Covariate drift vào vùng có ít dữ liệu | Huấn luyện lại trên dữ liệu gần đây có chứa những vùng đó |
| Prior drift | Hiệu chuẩn lại điểm số hoặc dịch ngưỡng quyết định |
| Concept drift | Thu thập nhãn mới trước. Huấn luyện lại trên nhãn cũ chỉ dạy lại khái niệm cũ. |

Sau khi huấn luyện lại, kiểm tra mô hình mới trên dữ liệu có nhãn gần nhất. Triển khai nó qua các giai đoạn shadow và canary từ bài thiết kế hệ thống. Rồi đặt lại cửa sổ tham chiếu về dữ liệu huấn luyện mới.

```figure
mlprod-drift-psi
```

## Tự tay làm

Code trong `code/main.py` chỉ dùng thư viện chuẩn.

### Bước 1: Bin theo phân vị và PSI

Các biên bin là các phân vị thập phân của tham chiếu. `bisect_right` tìm bin của từng giá trị.

```python
def quantile_edges(reference: list[float], bins: int = 10) -> list[float]:
    ordered = sorted(reference)
    n = len(ordered)
    edges = [ordered[min(n - 1, (n * i) // bins)] for i in range(1, bins)]
    return sorted(set(edges))


def psi_from_fractions(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total
```

### Bước 2: Thống kê KS và p-value của nó

Duyệt đồng thời hai mẫu đã sắp xếp. Tại mỗi giá trị phân biệt, tiến qua mọi bản sao của nó trong cả hai mẫu, rồi so sánh hai phần chia tích luỹ. Cách này xử lý trùng giá trị đúng.

```python
def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    a, b = sorted(sample_a), sorted(sample_b)
    n, m = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d
```

p-value dùng chuỗi cho phân phối Kolmogorov, `2 * sum((-1)^(k-1) * exp(-2 k^2 lambda^2))`, với `lambda = sqrt(n * m / (n + m)) * D`.

### Bước 3: Kiểm định chi-square

`chi2_sf` tính hàm sống chi-square thông qua hàm gamma bất toàn trên đã chuẩn hoá. Nó dùng chuỗi cho tham số nhỏ và phân số liên tục cho tham số lớn. `chi_square_test` lập bảng hai hàng, tính thống kê, và trả về p-value cùng Cramér's V.

```python
def chi_square_test(reference_counts: dict[str, int], current_counts: dict[str, int]) -> ChiSquareResult:
    categories = [c for c in sorted(set(reference_counts) | set(current_counts))
                  if reference_counts.get(c, 0) + current_counts.get(c, 0) > 0]
    rows = [[reference_counts.get(c, 0) for c in categories], [current_counts.get(c, 0) for c in categories]]
    row_totals = [sum(r) for r in rows]
    grand = sum(row_totals)
    if len(categories) < 2 or 0 in row_totals:
        return ChiSquareResult(0.0, 0, 1.0, 0.0)
    statistic = 0.0
    for col, _ in enumerate(categories):
        col_total = rows[0][col] + rows[1][col]
        for r in range(2):
            expected = row_totals[r] * col_total / grand
            statistic += (rows[r][col] - expected) ** 2 / expected
    dof = len(categories) - 1
    return ChiSquareResult(statistic, dof, chi2_sf(statistic, dof), math.sqrt(statistic / grand))
```

### Bước 4: Cửa sổ và bộ giám sát trôi lệch

`SlidingWindow` giữ `size` sự kiện cuối trong một `deque`. `DriftMonitor` tính các bin và số đếm tham chiếu một lần. Phương thức `check` của nó trả về PSI, `D` và p-value KS cho mỗi đặc trưng số, cùng kết quả chi-square cho mỗi đặc trưng phân loại.

### Bước 5: Chính sách cảnh báo

Chính sách biến mỗi bộ chỉ số thành một mức và đếm các cửa sổ trôi lệch liên tiếp theo từng tín hiệu:

```python
def level(self, metrics: dict[str, float]) -> str:
    if "psi" in metrics:
        if metrics["psi"] >= self.psi_drift or (metrics["ks"] >= self.ks_drift and metrics["ks_p"] < self.alpha):
            return "drift"
        return "warn" if metrics["psi"] >= self.psi_warn else "ok"
    if metrics["p"] < self.alpha and metrics["v"] >= self.min_cramers_v:
        return "drift"
    return "warn" if metrics["p"] < self.alpha else "ok"
```

`update` trả về `ALERT` ở cửa sổ hoàn tất chuỗi, và `resolved` ở cửa sổ sạch đầu tiên sau một báo động.

### Bước 6: Một luồng dữ liệu có cấy trôi lệch

Luồng có 9000 sự kiện thanh toán với một `amount`, một `channel` và một `label` gian lận. Bốn đoạn định nghĩa nó:

| Sự kiện | Amount | Tỷ lệ kênh (app, store, web) | Ranh giới nhãn | Thay đổi |
|---|---|---|---|---|
| 0 đến 2999 | trung bình 100, sd 20 | 35, 15, 50 phần trăm | 120 | Giai đoạn tham chiếu |
| 3000 đến 4999 | trung bình 118, sd 24 | 60, 10, 30 phần trăm | 120 | Covariate drift |
| 5000 đến 6499 | trung bình 100, sd 20 | 35, 15, 50 phần trăm | 120 | Trở lại bình thường |
| 6500 đến 8999 | trung bình 100, sd 20 | 35, 15, 50 phần trăm | 105 | Concept drift |

Mô hình là một ngưỡng trên `amount`, được fit trên 2000 sự kiện đầu. Cửa sổ giữ 500 sự kiện và trượt 250. Nhãn đến muộn 500 sự kiện.

```bash
python3 code/main.py
```

```console
  end    PSI   KS D   chi2 p    acc  levels (amount/channel/accuracy)  events
 3000  0.010  0.041  3.0e-01  0.932  ok/ok/ok
 3250  0.152  0.163  7.6e-06  0.924  drift/warn/ok
 3500  0.463  0.298  2.8e-26  0.936  drift/drift/ok                     ALERT amount
 3750  0.609  0.343  3.5e-35  0.942  drift/drift/ok                     ALERT channel
 ...
 5500  0.026  0.044  2.6e-01  0.898  ok/ok/ok                           resolved amount, resolved channel
 ...
 7250  0.040  0.048  7.1e-01  0.848  ok/ok/drift
 7500  0.022  0.039  6.1e-01  0.778  ok/ok/drift                        ALERT accuracy
```

Đọc kết quả theo ba phần:

- **Covariate drift.** Trôi lệch bắt đầu ở sự kiện 3000, và báo động `amount` theo sau ở 3500. Độ chính xác giữ trong khoảng 0.89 đến 0.95, vì quy tắc nhãn không đổi.
- **Phục hồi.** Đầu vào trở lại bình thường ở 5000, và cả hai báo động được giải quyết ở 5500.
- **Concept drift.** Trôi lệch bắt đầu ở 6500, và các bộ giám sát đầu vào vẫn im lặng vì đầu vào không đổi. Độ chính xác giảm từ 0.946 xuống khoảng 0.76 đến 0.79. Báo động độ chính xác đến ở 7500, 1000 sự kiện sau thay đổi, và độ trễ nhãn chiếm 500 trong số đó.

## Áp dụng

**scipy.** `scipy.stats` có cả hai kiểm định. Các thống kê khớp với code tự viết. Không cần scipy để chạy bài này.

```python
from scipy import stats

result = stats.ks_2samp(reference_amounts, current_amounts)
print(result.statistic, result.pvalue)

table = [[1000, 700, 300], [140, 290, 70]]
chi2 = stats.chi2_contingency(table)
print(chi2.statistic, chi2.pvalue, chi2.dof)
```

Chúng tôi đã đối chiếu `ks_statistic` và `chi_square_test` với scipy 1.18.1 trên các mẫu ngẫu nhiên. Thống kê KS, thống kê chi-square và p-value chi-square khớp tới độ chính xác dấu phẩy động. Trên 200 cặp mẫu ngẫu nhiên, các p-value KS khác của scipy tới 18 phần trăm giá trị của scipy. Các chênh lệch lớn nhất nằm ở những p-value rất nhỏ.

Chênh lệch đến từ phương pháp. Code của chúng ta dùng phân phối Kolmogorov giới hạn. Mặc định, `ks_2samp` dùng phương pháp chính xác khi cả hai mẫu có tối đa 10,000 giá trị. Chế độ tiệm cận của nó dùng phân phối mẫu hữu hạn với kích thước mẫu hiệu dụng đã làm tròn. Với hai danh mục, `chi2_contingency` mặc định áp dụng hiệu chỉnh liên tục Yates, nên các thống kê khác nhau ở đó.

**Thư viện giám sát.** Một số thư viện mã nguồn mở đóng gói các kiểm định này cùng báo cáo và dashboard:

- **Evidently** chọn một kiểm định trôi lệch cho mỗi cột. Với tham chiếu có 1000 hàng trở xuống, nó dùng kiểm định KS cho cột số và kiểm định chi-square cho cột phân loại. Với dữ liệu lớn hơn, nó dùng khoảng cách Wasserstein và phân kỳ Jensen-Shannon với ngưỡng 0.1. PSI là một trong những phương pháp khác mà bạn có thể chọn.
- **NannyML** ước lượng hiệu năng mô hình trước khi nhãn đến. Confidence-based Performance Estimation (CBPE) của nó dùng xác suất dự đoán của mô hình, mà phải được hiệu chuẩn tốt. Tài liệu NannyML nói rằng CBPE xử lý được covariate shift nhưng không chịu nổi concept drift. Luồng dữ liệu demo cho thấy vì sao: chỉ nhãn mới phơi ra concept drift.
- **TensorFlow Data Validation** so sánh các span dữ liệu liên tiếp. Nó dùng khoảng cách L-infinity cho đặc trưng phân loại và phân kỳ Jensen-Shannon xấp xỉ cho đặc trưng số. Nó cũng so sánh dữ liệu huấn luyện với dữ liệu phục vụ, điều mà bài feature store đề cập.

Code tự viết cho thấy mỗi công cụ trong số này tính ra cái gì. Các công cụ thêm lưu trữ, lập lịch, dashboard và nhiều kiểm định hơn nữa.

Với ứng dụng LLM, [bài về khả năng quan sát LLM](../../../17-infrastructure-and-production/13-llm-observability/docs/en.md) trong giai đoạn hạ tầng đề cập các công cụ tracing và đánh giá. Các ý tưởng trong bài này vẫn áp dụng cho đầu vào và đầu ra của những hệ thống đó.

## Đóng gói

Bài này tạo ra `outputs/skill-drift-monitoring.md`. Skill này lập kế hoạch giám sát cho một mô hình mới và phân loại một cảnh báo trôi lệch. Nó hỏi lớp nào đã phát, loại trôi lệch nào khớp với bằng chứng, và hành động đúng là chờ, hiệu chuẩn lại, thu thập nhãn hay huấn luyện lại.

## Bài tập

1. Chạy luồng dữ liệu với cửa sổ 200 và 2000 sự kiện. Ghi lại các cảnh cáo giả trước sự kiện 3000 và độ trễ báo động.
2. Đặt `n_tests=1`, `patience=1` và `min_cramers_v=0`. Đếm số báo động trong giai đoạn sạch từ 5000 đến 6500.
3. Thêm một đặc trưng có trung bình tăng 0.01 mỗi 100 sự kiện. Tìm xem tín hiệu nào, PSI hay KS, phản ứng trước.
4. Cài đặt khoảng cách Jensen-Shannon trên cùng các bin phân vị thập phân. So sánh nó với PSI trên mọi cửa sổ.
5. Thêm một bộ giám sát cho tỷ lệ dự đoán dương của mô hình. Giải thích vì sao nó vẫn im lặng trong suốt concept drift.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Data drift | "Dữ liệu đã thay đổi" | Thay đổi trong phân phối của đầu vào, dự đoán hoặc nhãn so với một cửa sổ tham chiếu |
| Covariate drift | "Kiểu người dùng mới" | `P(x)` thay đổi trong khi `P(y \| x)` giữ nguyên |
| Prior drift | "Nhiều dương tính hơn trước" | `P(y)` thay đổi trong khi `P(x \| y)` giữ nguyên, còn gọi là label shift |
| Concept drift | "Luật chơi đã thay đổi" | `P(y \| x)` thay đổi, nên cùng một đầu vào giờ có câu trả lời đúng khác |
| Reference window | "Dữ liệu mốc" | Một mẫu cố định, thường là dữ liệu huấn luyện, mà mọi cửa sổ hiện tại được so sánh với |
| PSI | "Điểm ổn định" | Tổng đối xứng trên các bin của (phần chia hiện tại trừ phần chia tham chiếu) nhân log tỷ số của chúng |
| KS statistic | "Kiểm định KS" | Khoảng cách lớn nhất giữa hai hàm phân phối tích luỹ thực nghiệm |
| Cramér's V | "Effect size cho danh mục" | Thống kê chi-square được co giãn về khoảng 0 đến 1 |
| Label delay | "Nhãn thực tế (ground truth) đến muộn" | Khoảng thời gian giữa một dự đoán và lúc kết quả thật của nó đến |
| Alert fatigue | "Quá nhiều page" | Người ta bỏ qua cảnh báo vì phần lớn chúng là báo động giả |

## Đọc thêm

- [Gama et al., A Survey on Concept Drift Adaptation (ACM Computing Surveys, 2014)](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation): real và virtual drift, các mẫu trôi lệch, và các phương pháp thích ứng
- [Rabanser, Günnemann, and Lipton, Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (NeurIPS 2019)](https://arxiv.org/abs/1810.11953): kiểm định hai mẫu để phát hiện dịch chuyển
- [Lu et al., Learning under Concept Drift: A Review (2020)](https://arxiv.org/abs/2004.05785): phát hiện, hiểu và thích ứng với trôi lệch
- [Yurdakul, Statistical Properties of Population Stability Index (Western Michigan University, 2018)](https://scholarworks.wmich.edu/dissertations/3208/): PSI hành xử thế nào theo kích thước mẫu và số bin
- [scipy.stats.ks_2samp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html): các phương pháp p-value chính xác và tiệm cận cho kiểm định KS hai mẫu
- [Evidently, How data drift detection works](https://docs.evidentlyai.com/metrics/explainer_drift): chọn kiểm định mặc định theo kiểu cột và kích thước dữ liệu
- [NannyML, Performance estimation](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html): CBPE và các giả định của nó

---

[← Bài trước: Thiết kế hệ thống ML](../../19-ml-system-design/docs/vi.md) · [Bài sau: Kho đặc trưng và lệch huấn luyện - phục vụ →](../../21-feature-stores-and-training-serving-skew/docs/vi.md)
