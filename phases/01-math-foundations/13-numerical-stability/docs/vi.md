# Ổn định số học

> Số dấu phẩy động (floating point) là một lớp trừu tượng bị rò rỉ. Nó sẽ cắn bạn trong lúc huấn luyện, và bạn sẽ không thấy nó đến.

**Type:** Build
**Language:** Python
**Prerequisites:** Giai đoạn 1, Bài 01-04
**Time:** ~120 phút

## Mục tiêu học tập

- Cài đặt softmax và log-sum-exp ổn định số học bằng thủ thuật trừ đi giá trị lớn nhất
- Nhận diện tràn trên (overflow), tràn dưới (underflow) và triệt tiêu thảm khốc (catastrophic cancellation) trong các phép tính dấu phẩy động
- Kiểm chứng gradient giải tích với gradient số bằng sai phân hữu hạn trung tâm
- Giải thích vì sao bfloat16 được ưu tiên hơn float16 khi huấn luyện và cách loss scaling ngăn gradient bị tràn dưới

## Vấn đề

Model của bạn huấn luyện ba tiếng, rồi loss thành NaN. Bạn thêm một lệnh print. Logits vẫn ổn ở bước 9,000. Đến bước 9,001 chúng là `inf`. Tới bước 9,002 thì mọi gradient là `nan` và quá trình huấn luyện chết.

Hoặc: model của bạn huấn luyện xong xuôi nhưng độ chính xác thấp hơn bài báo công bố 2%. Bạn kiểm tra mọi thứ. Kiến trúc khớp. Siêu tham số khớp. Dữ liệu khớp. Vấn đề là bài báo dùng float32 còn bạn dùng float16 mà không có scaling phù hợp. Ba mươi hai bit sai số làm tròn (rounding error) tích tụ đã âm thầm ăn mất độ chính xác của bạn.

Hoặc: bạn tự cài đặt cross-entropy loss từ đầu. Nó chạy được với logits nhỏ. Khi logits vượt 100, nó trả về `inf`. Softmax bị tràn trên vì `exp(100)` lớn hơn mức float32 biểu diễn được. Mọi framework ML đều xử lý việc này bằng một thủ thuật hai dòng. Bạn không biết thủ thuật đó tồn tại.

Ổn định số học (numerical stability) không phải chuyện lý thuyết. Nó là khác biệt giữa một lần huấn luyện thành công và một lần thất bại âm thầm. Mọi lỗi ML nghiêm trọng bạn sẽ gỡ rốt cuộc đều quy về số dấu phẩy động.

## Khái niệm

### IEEE 754: Cách máy tính lưu số thực

Máy tính lưu số thực dưới dạng giá trị dấu phẩy động theo chuẩn IEEE 754. Một float có ba phần: một bit dấu, một số mũ và một phần định trị (significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Phần định trị quyết định độ chính xác (precision) — tức có bao nhiêu chữ số có nghĩa. Số mũ quyết định miền giá trị — tức một số có thể lớn hay nhỏ đến đâu.

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 cho bạn khoảng 7 chữ số thập phân độ chính xác. Nghĩa là nó phân biệt được 1.0000001 và 1.0000002, nhưng không phân biệt được 1.00000001 và 1.00000002. Sau 7 chữ số, mọi thứ chỉ còn là nhiễu làm tròn.

float16 cho bạn khoảng 3 chữ số. Số lớn nhất nó biểu diễn được là 65,504. Con số đó nhỏ đến mức đáng lo cho ML, nơi logits, gradients và activations thường xuyên vượt mức này.

bfloat16 là câu trả lời của Google cho vấn đề miền giá trị của float16. Nó có số mũ 8 bit giống float32 (cùng miền giá trị, lên tới 3.4e38) nhưng chỉ 7 bit định trị (độ chính xác thấp hơn float16). Với việc huấn luyện mạng neural, miền giá trị quan trọng hơn độ chính xác, nên bfloat16 thường thắng.

### Vì sao 0.1 + 0.2 != 0.3

Số 0.1 không thể biểu diễn chính xác trong số dấu phẩy động nhị phân. Trong cơ số 2, nó là một phân số tuần hoàn:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 cắt cụt giá trị này còn 23 bit định trị. Giá trị được lưu xấp xỉ 0.100000001490116. Tương tự, 0.2 được lưu xấp xỉ 0.200000002980232. Tổng của chúng là 0.300000004470348, không phải 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Điều này quan trọng với ML vì:

1. Các phép so sánh loss như `if loss < threshold` có thể cho kết quả sai
2. Cộng dồn nhiều giá trị nhỏ (cập nhật gradient qua hàng nghìn bước) sẽ lệch dần khỏi tổng thật
3. Checksum và các bài kiểm tra tái lập sẽ hỏng nếu bạn so sánh float bằng `==`

Cách khắc phục: đừng bao giờ so sánh float bằng `==`. Dùng `abs(a - b) < epsilon` hoặc `math.isclose()`.

### Triệt tiêu thảm khốc

Khi bạn trừ hai số dấu phẩy động gần bằng nhau, các chữ số có nghĩa triệt tiêu lẫn nhau và bạn chỉ còn lại nhiễu làm tròn bị đẩy lên thành các chữ số hàng đầu.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Đó là sai số tương đối 19% chỉ từ một phép trừ. Trong ML, điều này xảy ra mỗi khi bạn:

- Tính phương sai của dữ liệu có trung bình lớn: `E[x^2] - E[x]^2` khi E[x] lớn
- Trừ những log-xác suất gần bằng nhau
- Tính gradient sai phân hữu hạn với epsilon quá nhỏ

Cách khắc phục: sắp xếp lại công thức để tránh trừ những số lớn gần bằng nhau. Với phương sai, dùng thuật toán Welford hoặc chuẩn hóa tâm dữ liệu trước. Với log-xác suất, hãy làm việc hoàn toàn trong không gian log.

### Tràn trên và tràn dưới

Tràn trên xảy ra khi kết quả quá lớn đến mức không biểu diễn được. Tràn dưới xảy ra khi kết quả quá nhỏ (gần 0 hơn số dương nhỏ nhất biểu diễn được).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Hàm `exp()` là nguồn tràn trên chính trong ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Hàm `log()` đi theo hướng ngược lại:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

Trong ML, `exp()` xuất hiện trong softmax, sigmoid và các phép tính xác suất. `log()` xuất hiện trong cross-entropy, log-likelihood và KL divergence. Tổ hợp `log(exp(x))` là một bãi mìn nếu không có đúng thủ thuật.

### Thủ thuật log-sum-exp

Tính trực tiếp `log(sum(exp(x_i)))` là nguy hiểm về mặt số học. Nếu có `x_i` nào lớn, `exp(x_i)` tràn trên. Nếu mọi `x_i` đều rất âm, từng `exp(x_i)` tràn dưới về 0 và `log(0)` là `-inf`.

Thủ thuật: trừ đi giá trị lớn nhất trước khi lấy số mũ.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Vì sao cách này hiệu quả: sau khi trừ `max(x)`, số mũ lớn nhất là `exp(0) = 1`. Không thể tràn trên. Ít nhất một số hạng trong tổng bằng 1, nên tổng ít nhất là 1, và `log(1) = 0`. Không thể tràn dưới thành `-inf`.

Chứng minh:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Đặt `c = max(x)` và tràn trên bị loại bỏ.

Thủ thuật này xuất hiện khắp nơi trong ML:
- Chuẩn hóa softmax
- Tính cross-entropy loss
- Cộng dồn log-xác suất trong mô hình chuỗi
- Hỗn hợp Gaussian
- Suy luận biến phân

### Vì sao softmax cần thủ thuật trừ max

Softmax biến logits thành xác suất:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Không có thủ thuật này, logits [100, 101, 102] gây tràn trên:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

Với thủ thuật, trừ đi max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Các xác suất giống hệt nhau. Phép tính an toàn. Đây không phải là tối ưu hóa. Đây là yêu cầu để có kết quả đúng.

### NaN và Inf: Phát hiện và phòng ngừa

`nan` (Not a Number) và `inf` (vô cực) lan truyền như virus qua phép tính. Một `nan` trong một cập nhật gradient làm weight thành `nan`, rồi làm mọi đầu ra sau đó thành `nan`. Quá trình huấn luyện chết chỉ trong một bước.

Cách `inf` xuất hiện:
- `exp()` của một số dương lớn
- Chia cho 0: `1.0 / 0.0`
- Tràn trên `float32` khi cộng dồn

Cách `nan` xuất hiện:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` của một số âm
- `log()` của một số âm
- Bất kỳ phép tính nào chứa `nan` đã có

Phát hiện:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Chiến lược phòng ngừa:

1. Chặn giá trị đầu vào của `exp()`: `exp(clamp(x, -80, 80))`
2. Cộng epsilon vào mẫu số: `x / (y + 1e-8)`
3. Cộng epsilon bên trong `log()`: `log(x + 1e-8)`
4. Dùng các cài đặt ổn định (log-sum-exp, softmax ổn định)
5. Cắt gradient (gradient clipping) để tránh weight bùng nổ
6. Kiểm tra `nan`/`inf` sau mỗi lượt forward khi gỡ lỗi

### Kiểm chứng gradient số

Gradient giải tích (từ lan truyền ngược) có thể có lỗi. Kiểm chứng gradient số xác minh chúng bằng cách tính gradient qua sai phân hữu hạn.

Công thức sai phân trung tâm:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Công thức này chính xác bậc O(h^2), tốt hơn nhiều so với sai phân tiến `(f(x+h) - f(x)) / h` chỉ đạt O(h).

Chọn h: quá lớn thì xấp xỉ sai. Quá nhỏ thì triệt tiêu thảm khốc phá hủy kết quả. `h = 1e-5` đến `1e-7` là điển hình.

Cách kiểm tra: tính sai khác tương đối giữa gradient giải tích và gradient số.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Quy tắc kinh nghiệm:
- relative_error < 1e-7: hoàn hảo, gradient đúng
- relative_error < 1e-5: chấp nhận được, nhiều khả năng đúng
- relative_error > 1e-3: có gì đó sai
- relative_error > 1: gradient sai hoàn toàn

Luôn kiểm tra gradient khi cài đặt một layer hay hàm loss mới. PyTorch cung cấp `torch.autograd.gradcheck()` cho việc này.

### Huấn luyện độ chính xác hỗn hợp

GPU hiện đại có phần cứng chuyên dụng (Tensor Cores) tính phép nhân ma trận float16 nhanh hơn float32 từ 2-8 lần. Huấn luyện độ chính xác hỗn hợp tận dụng điều này:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Vấn đề của huấn luyện thuần float16: gradient thường rất nhỏ (1e-8 hoặc nhỏ hơn). Float16 làm tràn dưới về 0 mọi giá trị dưới ~6e-8. Model của bạn ngừng học vì mọi cập nhật gradient đều bằng 0.

Cách khắc phục là loss scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling tự động điều chỉnh hệ số scale. Bắt đầu với giá trị lớn (65536). Nếu gradient tràn trên thành `inf`, giảm một nửa. Nếu qua N bước không tràn, tăng gấp đôi.

### bfloat16 vs float16: Vì sao bfloat16 thắng khi huấn luyện

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 có độ chính xác cao hơn (10 bit định trị so với 7) nhưng miền giá trị hạn chế (tối đa ~65,504). bfloat16 có độ chính xác thấp hơn nhưng miền giá trị giống float32 (tối đa ~3.4e38).

Khi huấn luyện mạng neural:

- Activations và logits thường xuyên vượt 65,504 trong những đợt tăng vọt khi huấn luyện. float16 tràn trên; bfloat16 xử lý được.
- Loss scaling là bắt buộc với float16 nhưng thường không cần với bfloat16 vì miền giá trị của nó bao trùm phổ độ lớn của gradient.
- bfloat16 chỉ là phép cắt cụt đơn giản của float32: bỏ 16 bit thấp nhất của phần định trị. Việc chuyển đổi rất đơn giản và không mất gì ở phần số mũ.

float16 được ưu tiên cho inference, nơi giá trị bị chặn trong biên và độ chính xác quan trọng hơn. bfloat16 được ưu tiên cho huấn luyện, nơi miền giá trị quan trọng hơn. Đây là lý do TPU và các GPU NVIDIA hiện đại (A100, H100) có hỗ trợ bfloat16 gốc.

### Cắt gradient (gradient clipping)

Gradient bùng nổ xảy ra khi gradient tăng theo cấp số nhân qua nhiều lớp (thường gặp ở RNN, mạng sâu và transformer). Chỉ một gradient lớn duy nhất có thể phá hỏng toàn bộ trọng số trong một bước.

Có hai kiểu cắt:

**Cắt theo giá trị:** kẹp từng phần tử gradient một cách độc lập.

```
grad = clamp(grad, -max_val, max_val)
```

Đơn giản nhưng có thể làm thay đổi hướng của vector gradient.

**Cắt theo norm:** co toàn bộ vector gradient sao cho norm của nó không vượt quá một ngưỡng.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Giữ nguyên hướng của gradient. Đây chính là việc `torch.nn.utils.clip_grad_norm_()` làm. Nó là lựa chọn tiêu chuẩn.

Giá trị thường dùng: `max_norm=1.0` cho transformer, `max_norm=0.5` cho RL, `max_norm=5.0` cho các mạng đơn giản hơn.

Cắt gradient không phải là một mẹo vặt. Đó là một cơ chế an toàn. Không có nó, chỉ một batch ngoại lai có thể tạo ra gradient đủ lớn để phá hỏng nhiều tuần huấn luyện.

### Các lớp chuẩn hóa như bộ ổn định số học

Batch normalization, layer normalization và RMS normalization thường được trình bày như những bộ điều chuẩn giúp quá trình huấn luyện hội tụ. Chúng cũng là những bộ ổn định số học.

Không có chuẩn hóa, các activation có thể tăng hoặc giảm theo cấp số nhân qua các lớp:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Chuẩn hóa căn lại tâm và co lại thang đo của activation ở mọi lớp:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (thường là 1e-5) ngăn phép chia cho 0 khi mọi activation giống nhau. Hai tham số học được `gamma` và `beta` cho phép mạng khôi phục bất kỳ thang đo nào mà nó cần.

Điều này giữ các giá trị trong một khoảng an toàn về mặt số học xuyên suốt mạng, ngăn cả tràn trên ở lượt lan truyền xuôi lẫn bùng nổ gradient ở lượt lan truyền ngược.

### Các lỗi số học thường gặp trong ML

**Lỗi: Loss là NaN sau vài epoch.**
Nguyên nhân: logits tăng quá lớn, softmax bị tràn. Hoặc learning rate quá cao và trọng số phân kỳ.
Cách sửa: dùng stable softmax (softmax ổn định) với phép trừ max, giảm learning rate, thêm cắt gradient.

**Lỗi: Loss đứng yên ở log(num_classes).**
Nguyên nhân: đầu ra của model gần như là xác suất đều. Thường có nghĩa là gradient đang biến mất hoặc model hoàn toàn không học được gì.
Cách sửa: kiểm tra nhãn dữ liệu có đúng không, xác minh hàm loss, kiểm tra các ReLU chết.

**Lỗi: Độ chính xác trên validation thấp hơn kỳ vọng 1-3%.**
Nguyên nhân: mixed precision (độ chính xác hỗn hợp) mà không có loss scaling (co giãn mất mát) đúng cách. Gradient bị underflow âm thầm đưa các cập nhật nhỏ về 0.
Cách sửa: bật loss scaling động, hoặc chuyển sang bfloat16.

**Lỗi: Norm gradient bằng 0.0 ở một số lớp.**
Nguyên nhân: neuron ReLU chết (mọi đầu vào đều âm), hoặc float16 bị underflow.
Cách sửa: dùng LeakyReLU hoặc GELU, dùng gradient scaling, kiểm tra khởi tạo trọng số.

**Lỗi: Model chạy trên một GPU nhưng cho kết quả khác trên GPU khác.**
Nguyên nhân: thứ tự tích lũy dấu phẩy động không tất định. Các phép rút gọn song song trên GPU cộng theo thứ tự khác nhau trên phần cứng khác nhau, và phép cộng dấu phẩy động không có tính kết hợp.
Cách sửa: chấp nhận khác biệt nhỏ (1e-6), hoặc đặt `torch.use_deterministic_algorithms(True)` và chấp nhận mất tốc độ.

**Lỗi: `exp()` trả về `inf` khi tính loss.**
Nguyên nhân: logits thô được đưa vào `exp()` mà không dùng thủ thuật trừ max.
Cách sửa: dùng `torch.nn.functional.log_softmax()` — hàm này triển khai log-sum-exp bên trong.

**Lỗi: Huấn luyện phân kỳ sau khi chuyển từ float32 sang float16.**
Nguyên nhân: float16 không biểu diễn được độ lớn gradient dưới 6e-8 hoặc activation trên 65,504.
Cách sửa: dùng mixed precision với loss scaling (AMP), hoặc dùng bfloat16 thay thế.

```figure
logsumexp-stability
```

## Tự tay làm

### Bước 1: Minh họa giới hạn độ chính xác của dấu phẩy động

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Bước 2: Triển khai softmax ngây thơ và softmax ổn định

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Bước 3: Triển khai log-sum-exp ổn định

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Bước 4: Triển khai cross-entropy ổn định

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Bước 5: Kiểm tra gradient

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Áp dụng

### Mô phỏng độ chính xác hỗn hợp

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Cắt gradient

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Phát hiện NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Xem `code/numerical.py` để có phần triển khai đầy đủ với mọi trường hợp biên được minh họa.

## Đóng gói

Bài học này tạo ra:
- `code/numerical.py` với softmax ổn định, log-sum-exp, cross-entropy, kiểm tra gradient và mô phỏng độ chính xác hỗn hợp
- `outputs/prompt-numerical-debugger.md` để chẩn đoán NaN/Inf và các vấn đề số học trong huấn luyện

Những triển khai ổn định này sẽ xuất hiện lại ở Phase 3 khi xây dựng vòng huấn luyện và ở Phase 4 khi triển khai các cơ chế attention.

## Bài tập

1. **Triệt tiêu thảm khốc (catastrophic cancellation).** Tính phương sai của [1000000.0, 1000001.0, 1000002.0] bằng công thức ngây thơ `E[x^2] - E[x]^2` trong float32. Sau đó tính bằng thuật toán trực tuyến của Welford. So sánh sai số với phương sai thật (0.6667).

2. **Truy tìm độ chính xác.** Tìm giá trị float32 dương nhỏ nhất `x` sao cho `1.0 + x == 1.0` trong Python. Đây là machine epsilon. Kiểm chứng rằng nó khớp với `numpy.finfo(numpy.float32).eps`.

3. **Các trường hợp biên của log-sum-exp.** Thử hàm `logsumexp_stable` của bạn với: (a) mọi giá trị bằng nhau, (b) một giá trị lớn hơn hẳn phần còn lại, (c) mọi giá trị rất âm (-1000). Kiểm chứng rằng nó cho kết quả đúng ở những chỗ phiên bản ngây thơ thất bại.

4. **Kiểm tra gradient cho một lớp mạng nơ-ron.** Triển khai một lớp tuyến tính đơn `y = Wx + b` và lượt lan truyền ngược giải tích của nó. Dùng `numerical_gradient` để kiểm chứng tính đúng đắn với ma trận trọng số 3x2.

5. **Thí nghiệm loss scaling.** Mô phỏng huấn luyện với float16: tạo gradient ngẫu nhiên trong khoảng [1e-9, 1e-3], chuyển sang float16 và đo tỉ lệ bao nhiêu phần trở thành 0. Sau đó áp dụng loss scaling (nhân với 1024), chuyển sang float16, co lại thang đo và đo lại tỉ lệ bằng 0.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| IEEE 754 | "Chuẩn float" | Chuẩn quốc tế định nghĩa các định dạng dấu phẩy động nhị phân, quy tắc làm tròn và các giá trị đặc biệt (inf, nan). Mọi CPU và GPU hiện đại đều triển khai nó. |
| Machine epsilon | "Giới hạn độ chính xác" | Giá trị e nhỏ nhất sao cho 1.0 + e != 1.0 trong một định dạng float cho trước. Với float32, giá trị này khoảng 1.19e-7. |
| Catastrophic cancellation | "Mất độ chính xác do phép trừ" | Khi trừ hai số dấu phẩy động gần bằng nhau, các chữ số có nghĩa triệt tiêu lẫn nhau và nhiễu làm tròn chi phối kết quả. |
| Overflow | "Số quá lớn" | Kết quả vượt quá giá trị lớn nhất biểu diễn được và trở thành inf. exp(89) tràn float32. |
| Underflow | "Số quá nhỏ" | Kết quả gần 0 hơn số dương nhỏ nhất biểu diễn được và trở thành 0.0. exp(-104) dưới ngưỡng float32. |
| Log-sum-exp trick | "Trừ max trước đã" | Tính log(sum(exp(x))) bằng cách tách nhân tử exp(max(x)) để tránh tràn trên và tràn dưới. Được dùng trong softmax, cross-entropy và các phép toán log-xác suất. |
| Stable softmax | "Softmax không nổ" | Trừ max(logits) trước khi lũy thừa. Kết quả giống hệt về mặt số học, không thể tràn. |
| Gradient checking | "Kiểm tra lại backprop" | So sánh gradient giải tích từ lan truyền ngược với gradient số từ sai phân hữu hạn để phát hiện lỗi triển khai. |
| Mixed precision | "Float16 lượt xuôi, float32 lượt ngược" | Dùng float độ chính xác thấp cho các phép toán cần tốc độ và float độ chính xác cao cho các phép toán nhạy cảm về số học. Tốc độ thường nhanh hơn 2-3 lần. |
| Loss scaling | "Ngăn gradient underflow" | Nhân loss với một hằng số lớn trước khi lan truyền ngược để gradient nằm trong khoảng biểu diễn của float16, rồi chia cho cùng hằng số đó trước khi cập nhật trọng số. |
| bfloat16 | "Dấu phẩy động kiểu não" | Định dạng 16-bit của Google với 8 bit mũ (cùng khoảng với float32) và 7 bit định trị (ít chính xác hơn float16). Được ưu tiên cho huấn luyện. |
| Gradient clipping | "Chặn norm gradient" | Co vector gradient sao cho norm của nó không vượt quá một ngưỡng. Ngăn gradient bùng nổ phá hỏng trọng số. |
| NaN | "Not a Number" | Giá trị float đặc biệt sinh ra từ các phép toán không xác định (0/0, inf-inf, sqrt(-1)). Nó lan truyền qua mọi phép số học về sau. |
| Inf | "Vô cùng" | Giá trị float đặc biệt sinh ra từ tràn trên hoặc chia cho 0. Có thể kết hợp để tạo NaN (inf - inf, inf * 0). |
| Numerical gradient | "Đạo hàm kiểu vét cạn" | Xấp xỉ đạo hàm bằng cách tính f(x+h) và f(x-h) rồi chia cho 2h. Chậm nhưng đáng tin cậy để kiểm chứng. |

## Đọc thêm

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- tài liệu tham khảo kinh điển, dày đặc nhưng đầy đủ
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- bài báo NVIDIA giới thiệu loss scaling cho huấn luyện float16
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- hướng dẫn thực hành về mixed precision trong PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- vì sao Google chọn định dạng này cho TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- thuật toán giảm sai số làm tròn trong các phép tổng dấu phẩy động
