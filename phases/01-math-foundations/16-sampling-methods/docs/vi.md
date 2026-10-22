# Các phương pháp lấy mẫu

> Lấy mẫu (sampling) là cách AI khám phá không gian các khả năng.

**Type:** Build
**Language:** Python
**Prerequisites:** Giai đoạn 1, Bài 06-07 (Xác suất, Định lý Bayes)
**Time:** ~120 phút

## Mục tiêu học tập

- Tự tay viết lấy mẫu CDF nghịch đảo, lấy mẫu loại bỏ (rejection sampling) và lấy mẫu quan trọng, chỉ dùng các số ngẫu nhiên đều
- Xây dựng temperature (nhiệt độ), top-k và top-p (nucleus sampling, tức lấy mẫu hạt nhân) để sinh token cho mô hình ngôn ngữ
- Giải thích thủ thuật tái tham số hóa và vì sao nó cho phép lan truyền ngược xuyên qua phép lấy mẫu trong VAE
- Chạy MCMC Metropolis-Hastings để lấy mẫu từ một phân phối đích chưa chuẩn hóa

## Vấn đề

Một mô hình ngôn ngữ xử lý xong prompt của bạn và tạo ra một vector gồm 50.000 logits. Mỗi logit ứng với một token trong từ vựng của nó. Giờ nó phải chọn một. Chọn thế nào?

Nếu nó luôn chọn token có xác suất cao nhất, mọi câu trả lời đều giống hệt nhau. Tất định. Nhàm chán. Nếu nó chọn hoàn toàn ngẫu nhiên, đầu ra là những chuỗi vô nghĩa. Câu trả lời nằm đâu đó giữa hai thái cực này, và cái "đâu đó" ấy do lấy mẫu điều khiển.

Lấy mẫu không chỉ giới hạn ở sinh văn bản. Học tăng cường ước lượng gradient của chính sách bằng cách lấy mẫu các quỹ đạo. VAE học các biểu diễn ẩn bằng cách lấy mẫu từ những phân phối (distribution) đã học rồi lan truyền ngược xuyên qua phần ngẫu nhiên đó. Mô hình khuếch tán sinh ảnh bằng cách lấy mẫu nhiễu rồi khử nhiễu lặp đi lặp lại. Phương pháp Monte Carlo ước lượng những tích phân không có nghiệm dạng đóng. Thuật toán MCMC khám phá các phân phối hậu nghiệm nhiều chiều mà không thể liệt kê hết.

Mọi hệ thống AI sinh tạo đều là một hệ thống lấy mẫu. Chiến lược lấy mẫu quyết định chất lượng, độ đa dạng và khả năng điều khiển của đầu ra. Bài này tự tay xây dựng mọi phương pháp lấy mẫu chủ chốt, bắt đầu từ các số ngẫu nhiên đều và kết thúc bằng những kỹ thuật đang vận hành các LLM và mô hình sinh tạo hiện đại.

## Khái niệm

### Vì sao lấy mẫu quan trọng

Lấy mẫu xuất hiện ở bốn vai trò nền tảng trong AI và học máy:

**Sinh (generation).** Mô hình ngôn ngữ, mô hình khuếch tán và GAN đều tạo đầu ra bằng cách lấy mẫu. Thuật toán lấy mẫu điều khiển trực tiếp độ sáng tạo, độ mạch lạc và độ đa dạng. Temperature, top-k và nucleus sampling là những núm vặn mà kỹ sư chỉnh hằng ngày.

**Huấn luyện (training).** Hạ gradient ngẫu nhiên lấy mẫu các mini-batch. Dropout lấy mẫu neuron để tắt đi. Tăng cường dữ liệu lấy mẫu các phép biến đổi ngẫu nhiên. Lấy mẫu quan trọng đánh trọng số lại các mẫu để giảm phương sai gradient trong học tăng cường (PPO, TRPO).

**Ước lượng (estimation).** Nhiều đại lượng trong ML không có nghiệm dạng đóng. Kỳ vọng tổn thất trên một phân phối dữ liệu, hàm phân hoạch của mô hình dựa trên năng lượng, bằng chứng trong suy diễn Bayes. Ước lượng Monte Carlo xấp xỉ tất cả những thứ đó bằng cách lấy trung bình trên các mẫu.

**Khám phá (exploration).** Thuật toán MCMC khám phá các phân phối hậu nghiệm trong suy diễn Bayes. Chiến lược tiến hóa lấy mẫu các nhiễu loạn tham số. Thompson sampling cân bằng giữa khám phá và khai thác trong bài toán bandit.

Thách thức cốt lõi: bạn chỉ lấy mẫu trực tiếp được từ những phân phối đơn giản (đều, chuẩn). Với mọi thứ còn lại, bạn cần một phương pháp biến các mẫu đơn giản thành mẫu từ phân phối đích của mình.

### Lấy mẫu ngẫu nhiên đều

Mọi phương pháp lấy mẫu đều bắt đầu từ đây. Một bộ sinh số ngẫu nhiên đều tạo ra các giá trị trong [0, 1) sao cho mọi khoảng con có cùng độ dài đều có xác suất bằng nhau.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

Để lấy mẫu đều từ một tập rời rạc gồm n phần tử, sinh U rồi trả về floor(n * U). Để lấy mẫu từ một khoảng liên tục [a, b], tính a + (b - a) * U.

Điểm mấu chốt: một số ngẫu nhiên đều duy nhất chứa đúng lượng ngẫu nhiên cần thiết để tạo ra một mẫu từ bất kỳ phân phối nào. Mẹo nằm ở việc tìm ra phép biến đổi đúng.

### Phương pháp CDF nghịch đảo (lấy mẫu biến đổi nghịch đảo)

Hàm phân phối tích lũy (CDF) ánh xạ giá trị sang xác suất:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

CDF nghịch đảo ánh xạ xác suất trở lại giá trị. Nếu U ~ Uniform(0, 1), thì X = F_inverse(U) tuân theo phân phối đích.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Ví dụ với phân phối mũ:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Cách này hoạt động hoàn hảo khi bạn viết được F_inverse ở dạng đóng. Với phân phối chuẩn, không có CDF nghịch đảo dạng đóng, nên ta dùng các phương pháp khác (Box-Muller, hoặc xấp xỉ số).

**Phiên bản rời rạc:** Với phân phối rời rạc (categorical distribution), hãy dựng CDF thành tổng tích lũy, sinh U, rồi tìm chỉ số đầu tiên mà tổng tích lũy vượt U. Đây là cách `sample_categorical` hoạt động ở Bài 06.

### Lấy mẫu loại bỏ

Khi bạn không nghịch đảo được CDF nhưng tính được PDF đích sai khác một hằng số, lấy mẫu loại bỏ sẽ dùng được.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Ràng buộc M càng chặt thì tỉ lệ chấp nhận càng cao. Ở số chiều thấp (1-3), lấy mẫu loại bỏ hoạt động tốt. Ở số chiều cao, tỉ lệ chấp nhận giảm theo hàm mũ vì phần lớn thể tích của phân phối đề xuất bị loại. Đây chính là lời nguyền số chiều đối với lấy mẫu loại bỏ.

**Ví dụ: lấy mẫu từ phân phối chuẩn bị cắt cụt.** Dùng một phân phối đề xuất đều trên khoảng bị cắt cụt. Bao hình M là giá trị lớn nhất của PDF chuẩn trên khoảng đó.

**Ví dụ: lấy mẫu từ nửa đường tròn.** Đề xuất đều trong hình chữ nhật bao quanh. Chấp nhận nếu điểm rơi vào bên trong nửa đường tròn. Đây là cách Monte Carlo tính pi: tỉ lệ chấp nhận bằng tỉ lệ diện tích pi/4.

### Lấy mẫu quan trọng

Đôi khi bạn không cần các mẫu từ phân phối đích p(x). Bạn cần ước lượng một kỳ vọng theo p(x), mà bạn lại có sẵn các mẫu từ một phân phối khác q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Điều này rất quan trọng trong học tăng cường. Trong PPO (Proximal Policy Optimization), bạn thu các quỹ đạo dưới chính sách cũ pi_old nhưng muốn tối ưu chính sách mới pi_new. Trọng số quan trọng là pi_new(a|s) / pi_old(a|s). PPO cắt các trọng số này để ngăn chính sách mới lệch quá xa chính sách cũ.

Phương sai của ước lượng lấy mẫu quan trọng phụ thuộc vào mức độ giống nhau giữa q và p. Nếu q rất khác p, một vài mẫu sẽ nhận trọng số khổng lồ và chi phối ước lượng. Lấy mẫu quan trọng tự chuẩn hóa chia cho tổng các trọng số để giảm vấn đề này:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Ước lượng Monte Carlo

Ước lượng Monte Carlo xấp xỉ tích phân bằng cách lấy trung bình trên các mẫu ngẫu nhiên. Luật số lớn bảo đảm sự hội tụ.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Tốc độ sai số không phụ thuộc số chiều. Đó là lý do các phương pháp Monte Carlo chiếm ưu thế ở số chiều cao, nơi tích phân dựa trên lưới là bất khả thi.

**Ước lượng pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Ước lượng kỳ vọng:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC dựng một chuỗi Markov có phân phối dừng chính là phân phối đích p(x). Sau đủ số bước, các mẫu từ chuỗi là (xấp xỉ) mẫu từ p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Với các đề xuất đối xứng (q(x'|x) = q(x|x')), tỉ số rút gọn thành p(x')/p(x). Đây là thuật toán Metropolis nguyên thủy.

**Vì sao nó hoạt động.** Quy tắc chấp nhận bảo đảm cân bằng chi tiết: xác suất đang ở x rồi chuyển sang x' bằng xác suất đang ở x' rồi chuyển sang x. Cân bằng chi tiết kéo theo việc p(x) là phân phối dừng của chuỗi.

**Những điểm cần lưu ý khi thực hành:**
- Burn-in: bỏ các mẫu đầu trước khi chuỗi đạt trạng thái cân bằng
- Thinning: giữ mỗi mẫu thứ k để giảm tự tương quan
- Độ lớn bước đề xuất: quá nhỏ thì chuỗi di chuyển chậm (tỉ lệ chấp nhận cao, khám phá chậm); quá lớn thì phần lớn đề xuất bị loại (tỉ lệ chấp nhận thấp, kẹt tại chỗ)
- Tỉ lệ chấp nhận tối ưu cho đề xuất Gauss ở số chiều cao xấp xỉ 0.234

### Lấy mẫu Gibbs

Lấy mẫu Gibbs là một trường hợp riêng của MCMC cho các phân phối nhiều biến. Thay vì đề xuất một bước dịch chuyển trên tất cả các chiều cùng lúc, nó cập nhật từng biến một theo phân phối điều kiện của biến đó.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Lấy mẫu Gibbs đòi hỏi bạn lấy mẫu được từ từng phân phối điều kiện p(x_i | x_{-i}). Điều này đơn giản với nhiều mô hình:
- Mạng Bayes: các phân phối điều kiện suy ra từ cấu trúc đồ thị
- Hỗn hợp Gauss: các phân phối điều kiện là Gauss
- Mô hình Ising: phân phối điều kiện của mỗi spin chỉ phụ thuộc các lân cận của nó

Tỉ lệ chấp nhận luôn bằng 1 (mọi đề xuất đều được chấp nhận) vì lấy mẫu từ đúng phân phối điều kiện tự động thỏa mãn cân bằng chi tiết.

**Hạn chế.** Khi các biến tương quan mạnh, lấy mẫu Gibbs trộn chậm vì cập nhật từng biến một không thể tạo ra những bước dịch chuyển chéo lớn xuyên qua phân phối.

### Lấy mẫu theo nhiệt độ (dùng trong LLM)

Mô hình ngôn ngữ xuất ra các logits z_1, ..., z_V cho mỗi token trong từ vựng. Softmax chuyển chúng thành xác suất. Nhiệt độ tái co giãn các logits trước softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Vì sao nó hoạt động.** Chia logits cho T < 1 khuếch đại khác biệt giữa các logits. Nếu z_1 = 2 và z_2 = 1, chia cho T = 0.5 được z_1/T = 4 và z_2/T = 2, khiến khoảng cách lớn hơn. Sau softmax, token có logit cao nhất nhận phần chia lớn hơn hẳn.

**Trong thực tế:**
- T = 0.0: giải mã tham lam, tốt nhất cho hỏi đáp dựa trên sự kiện
- T = 0.3-0.7: hơi sáng tạo, tốt cho sinh mã
- T = 0.7-1.0: cân bằng, tốt cho hội thoại thông thường
- T = 1.0-1.5: viết sáng tạo, động não ý tưởng
- T > 1.5: ngày càng ngẫu nhiên, hiếm khi hữu ích

Nhiệt độ không thay đổi những token nào có thể xuất hiện. Nó thay đổi khối xác suất được chia cho mỗi token.

### Lấy mẫu top-k

Lấy mẫu top-k giới hạn tập ứng viên vào k token có xác suất cao nhất, sau đó chuẩn hóa lại và lấy mẫu từ tập bị giới hạn đó.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k ngăn mô hình chọn những token cực kỳ khó xảy ra (lỗi đánh máy, chuỗi vô nghĩa) vốn tồn tại ở phần đuôi dài của phân phối từ vựng. Vấn đề: k là cố định bất kể ngữ cảnh. Khi mô hình tự tin (một token chiếm 95% xác suất), k = 40 vẫn cho phép 39 lựa chọn thay thế. Khi mô hình không chắc chắn (xác suất trải trên 1000 token), k = 40 cắt bỏ những lựa chọn hợp lý.

### Lấy mẫu top-p (hạt nhân)

Lấy mẫu top-p điều chỉnh linh hoạt kích thước tập ứng viên. Thay vì giữ một số token cố định, nó giữ tập token nhỏ nhất mà tổng xác suất tích lũy vượt p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Khi mô hình tự tin, lấy mẫu hạt nhân giữ ít token (có thể 2-3). Khi mô hình không chắc chắn, nó giữ nhiều token (có thể 200). Chính hành vi thích ứng này khiến lấy mẫu hạt nhân thường sinh văn bản tốt hơn top-k.

**Các kết hợp thường dùng:**
- Nhiệt độ 0.7 + top-p 0.9: thiết lập đa dụng tốt
- Nhiệt độ 0.0 (tham lam): tốt nhất cho các tác vụ tất định
- Nhiệt độ 1.0 + top-k 50: thiết lập trong bài báo gốc của Fan et al. (2018)

Có thể kết hợp top-k và top-p. Áp dụng top-k trước, rồi top-p trên tập còn lại.

### Thủ thuật tái tham số hóa (dùng trong VAE)

Variational autoencoder (VAE) học bằng cách mã hóa đầu vào thành một phân phối trong không gian ẩn, lấy mẫu từ phân phối đó, rồi giải mã mẫu trở lại. Vấn đề: bạn không thể lan truyền ngược xuyên qua một phép lấy mẫu.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Thủ thuật tái tham số hóa tách phần ngẫu nhiên khỏi các tham số:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Cách này hoạt động vì N(mu, sigma^2) có cùng phân phối với mu + sigma * N(0, 1). Điểm mấu chốt: đưa phần ngẫu nhiên về một nguồn không có tham số (epsilon), rồi biểu diễn mẫu như một phép biến đổi khả vi của các tham số.

**Trong vòng lặp huấn luyện VAE:**
1. Encoder xuất ra mu và log(sigma^2) cho mỗi đầu vào
2. Lấy mẫu epsilon ~ N(0, 1)
3. Tính z = mu + sigma * epsilon
4. Giải mã z để tái tạo đầu vào
5. Lan truyền ngược qua các bước 4, 3, 2, 1 (khả thi vì bước 3 khả vi)

Không có thủ thuật tái tham số hóa, VAE không thể huấn luyện bằng lan truyền ngược tiêu chuẩn. Chính phát hiện này đã làm VAE trở nên khả dụng.

### Gumbel-Softmax (Lấy mẫu phân phối rời rạc khả vi)

Thủ thuật tái tham số hóa hoạt động với các phân phối liên tục (Gaussian). Với phân phối rời rạc categorical, ta cần một cách tiếp cận khác. Gumbel-Softmax cung cấp một xấp xỉ khả vi cho việc lấy mẫu categorical.

**Thủ thuật Gumbel-Max (không khả vi):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (xấp xỉ khả vi):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax tạo ra một dạng nới lỏng liên tục của một mẫu rời rạc. Đầu ra là một vector xác suất (soft one-hot) thay vì hard one-hot. Gradient chảy qua softmax. Trong lượt lan truyền xuôi khi huấn luyện, bạn có thể dùng bộ ước lượng "straight-through": dùng argmax cứng cho lượt xuôi nhưng dùng gradient mềm của Gumbel-Softmax cho lượt ngược.

**Ứng dụng:**
- Biến tiềm ẩn rời rạc trong VAE
- Tìm kiếm kiến trúc neural (chọn các phép toán rời rạc)
- Cơ chế attention cứng
- Học tăng cường với hành động rời rạc

### Lấy mẫu phân tầng

Lấy mẫu Monte Carlo chuẩn có thể để lại những khoảng trống trong không gian mẫu do ngẫu nhiên. Lấy mẫu phân tầng buộc phải phủ đều bằng cách chia không gian thành các tầng rồi lấy mẫu từ mỗi tầng.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Lấy mẫu phân tầng luôn có phương sai nhỏ hơn hoặc bằng so với Monte Carlo chuẩn:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Ứng dụng:**
- Tích phân số (quasi-Monte Carlo)
- Chia tách dữ liệu huấn luyện (đảm bảo cân bằng lớp trong mỗi fold)
- Lấy mẫu quan trọng có phân tầng (kết hợp cả hai kỹ thuật)
- NeRF (Neural Radiance Fields) dùng lấy mẫu phân tầng dọc theo các tia camera

### Liên hệ với mô hình khuếch tán

Mô hình khuếch tán sinh ảnh thông qua một quá trình lấy mẫu. Quá trình xuôi thêm nhiễu Gaussian vào ảnh qua T bước cho đến khi ảnh trở thành nhiễu thuần. Quá trình ngược học cách khử nhiễu, khôi phục ảnh gốc từng bước một.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Mối liên hệ với các phương pháp trong bài này:
- Mỗi bước khử nhiễu dùng thủ thuật tái tham số hóa (lấy mẫu nhiễu, áp dụng phép biến đổi tất định)
- Lịch nhiễu {alpha_t} điều khiển một dạng temperature annealing (ủ nhiệt độ)
- Việc huấn luyện dùng ước lượng Monte Carlo để xấp xỉ ELBO (cận dưới bằng chứng)
- Lấy mẫu tổ tiên (ancestral sampling) trong mô hình khuếch tán là một chuỗi Markov (mỗi bước chỉ phụ thuộc vào trạng thái hiện tại)

Toàn bộ quá trình sinh ảnh là lấy mẫu lặp: bắt đầu từ nhiễu, và ở mỗi bước, lấy mẫu một phiên bản ít nhiễu hơn một chút, có điều kiện trên mô hình khử nhiễu đã học.

```figure
monte-carlo-pi
```

## Tự tay làm

### Bước 1: Lấy mẫu đều và lấy mẫu nghịch đảo CDF

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Sinh 10,000 mẫu phân phối mũ và kiểm tra rằng giá trị trung bình là 1/lambda.

### Bước 2: Lấy mẫu loại bỏ

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Dùng lấy mẫu loại bỏ để rút mẫu từ phân phối chuẩn bị cắt (truncated normal). Kiểm tra hình dạng bằng cách vẽ histogram của các mẫu.

### Bước 3: Lấy mẫu quan trọng

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Ước lượng E[X^2] dưới phân phối chuẩn bằng một phân phối đề xuất đều. So sánh với đáp án đã biết (mu^2 + sigma^2).

### Bước 4: Ước lượng pi bằng Monte Carlo

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Bước 5: MCMC Metropolis-Hastings

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Lấy mẫu từ phân phối hai đỉnh (hỗn hợp hai Gaussian). Trực quan hóa quỹ đạo của chuỗi.

### Bước 6: Lấy mẫu Gibbs

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Bước 7: Lấy mẫu theo temperature

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Cho thấy temperature thay đổi phân phối đầu ra như thế nào với một tập logits của token.

### Bước 8: Lấy mẫu top-k và top-p

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Bước 9: Thủ thuật tái tham số hóa

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Chứng minh rằng gradient chảy qua mẫu đã tái tham số hóa nhưng không chảy qua việc lấy mẫu trực tiếp.

### Bước 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Cho thấy việc giảm temperature khiến đầu ra tiến gần tới một vector one-hot.

Bản cài đặt đầy đủ cùng toàn bộ phần trực quan hóa nằm trong `code/sampling.py`.

## Áp dụng

Với NumPy và SciPy, đây là các phiên bản production:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Với MCMC ở quy mô lớn, hãy dùng các thư viện chuyên dụng:
- PyMC: mô hình hóa Bayesian đầy đủ với NUTS (HMC thích ứng)
- emcee: bộ lấy mẫu MCMC theo tập hợp (ensemble)
- NumPyro/JAX: MCMC tăng tốc bằng GPU

Bạn đã tự xây dựng những thứ này từ đầu. Giờ bạn đã biết các lệnh gọi thư viện đang làm gì.

## Bài tập

1. Cài đặt lấy mẫu nghịch đảo CDF cho phân phối Cauchy. CDF là F(x) = 0.5 + arctan(x)/pi. Sinh 10,000 mẫu và vẽ histogram đối chiếu với PDF thật. Chú ý phần đuôi nặng (các giá trị cực trị nằm xa tâm).

2. Dùng lấy mẫu loại bỏ để sinh mẫu từ phân phối Beta(2, 5) với phân phối đề xuất Uniform(0, 1). Vẽ các mẫu được chấp nhận đối chiếu với PDF Beta thật. Tỷ lệ chấp nhận lý thuyết là bao nhiêu?

3. Ước lượng tích phân của sin(x) từ 0 đến pi bằng Monte Carlo với 1,000, 10,000 và 100,000 mẫu. So sánh sai số ở từng mức. Kiểm chứng rằng sai số co lại theo O(1/sqrt(N)).

4. Cài đặt Metropolis-Hastings để lấy mẫu từ phân phối 2D p(x, y) tỷ lệ với exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Vẽ các mẫu và quỹ đạo của chuỗi. Thử nghiệm với các độ lệch chuẩn đề xuất khác nhau.

5. Xây dựng một demo sinh văn bản hoàn chỉnh: cho một từ vựng 10 từ kèm logits, sinh các chuỗi 20 token bằng (a) greedy (giải mã tham lam), (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. So sánh độ đa dạng của đầu ra qua 5 lần chạy.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Sampling | "Rút các giá trị ngẫu nhiên" | "Sinh giá trị theo một phân phối xác suất. Cơ chế đằng sau mọi AI sinh" |
| Uniform distribution | "Mọi giá trị đều như nhau" | "Mỗi giá trị trong [a, b] có mật độ xác suất bằng nhau 1/(b-a). Điểm khởi đầu cho mọi phương pháp lấy mẫu" |
| Inverse CDF | "Biến đổi xác suất" | "F_inverse(U) biến một mẫu đều thành mẫu từ bất kỳ phân phối nào có CDF đã biết. Chính xác và hiệu quả" |
| Rejection sampling | "Đề xuất rồi chấp nhận/loại" | "Sinh từ một phân phối đề xuất đơn giản, chấp nhận với xác suất tỷ lệ với tỷ số target/proposal. Chính xác nhưng lãng phí mẫu" |
| Importance sampling | "Đánh trọng số lại các mẫu" | "Ước lượng kỳ vọng dưới p(x) bằng các mẫu từ q(x) bằng cách gán trọng số p(x)/q(x) cho mỗi mẫu. Cốt lõi của PPO trong RL" |
| Monte Carlo | "Lấy trung bình các mẫu ngẫu nhiên" | "Xấp xỉ tích phân bằng trung bình mẫu. Sai số O(1/sqrt(N)) bất kể số chiều" |
| MCMC | "Bước đi ngẫu nhiên rồi hội tụ" | "Xây dựng một chuỗi Markov có phân phối dừng là phân phối mục tiêu. Metropolis-Hastings là thuật toán nền tảng" |
| Metropolis-Hastings | "Chấp nhận khi đi lên, đôi khi cả khi đi xuống" | "Đề xuất bước di chuyển, chấp nhận dựa trên tỷ số mật độ. Cân bằng chi tiết đảm bảo hội tụ về phân phối mục tiêu" |
| Gibbs sampling | "Mỗi lần một biến" | "Cập nhật từng biến theo phân phối có điều kiện trong khi giữ nguyên các biến khác. Tỷ lệ chấp nhận 100%" |
| Temperature | "Núm chỉnh độ tự tin" | "Chia logits cho T trước softmax. T<1 làm nhọn (tự tin hơn), T>1 làm phẳng (đa dạng hơn)" |
| Top-k sampling | "Giữ k cái tốt nhất" | "Đặt về 0 mọi token trừ k token có xác suất cao nhất, chuẩn hóa lại, rồi lấy mẫu. Kích thước tập ứng viên cố định" |
| Nucleus sampling (top-p) | "Giữ những cái có khả năng" | "Giữ tập nhỏ nhất các token có tổng xác suất tích lũy vượt p. Kích thước tập ứng viên thích ứng" |
| Reparameterization trick | "Đưa phần ngẫu nhiên ra ngoài" | "Viết z = mu + sigma * epsilon với epsilon ~ N(0,1). Làm cho việc lấy mẫu khả vi. Thiết yếu cho huấn luyện VAE" |
| Gumbel-Softmax | "Lấy mẫu categorical mềm" | "Xấp xỉ khả vi của lấy mẫu categorical dùng nhiễu Gumbel + softmax với temperature" |
| Stratified sampling | "Phủ đều cưỡng bức" | "Chia không gian mẫu thành các tầng, lấy mẫu từ mỗi tầng. Phương sai luôn nhỏ hơn Monte Carlo ngây thơ" |
| Burn-in | "Giai đoạn khởi động" | "Các mẫu MCMC ban đầu bị bỏ đi trước khi chuỗi đạt phân phối dừng" |
| Detailed balance | "Điều kiện thuận nghịch" | "p(x) * T(x->y) = p(y) * T(y->x). Điều kiện đủ để p là phân phối dừng của một chuỗi Markov" |
| Diffusion sampling | "Khử nhiễu lặp" | "Sinh dữ liệu bằng cách bắt đầu từ nhiễu và áp dụng các bước khử nhiễu đã học. Mỗi bước là một phép lấy mẫu có điều kiện" |

## Đọc thêm

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - bài hướng dẫn chi tiết về nền tảng MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - bài báo gốc về Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - bài báo về lấy mẫu nucleus (top-p)
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - bài báo VAE giới thiệu thủ thuật tái tham số hóa
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM liên hệ việc lấy mẫu với sinh ảnh

---

[← Bài trước: Thống kê cho Học máy](../../15-statistics-for-ml/docs/vi.md) · [Bài sau: Hệ phương trình tuyến tính →](../../17-linear-systems/docs/vi.md)
