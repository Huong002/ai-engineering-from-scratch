# Giới thiệu về JAX

> PyTorch biến đổi tensor tại chỗ. TensorFlow dựng đồ thị. JAX biên dịch các hàm thuần. Điều cuối cùng đó thay đổi cách bạn nghĩ về deep learning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 03, Bài 01-10, NumPy cơ bản
**Time:** ~90 phút

## Mục tiêu học tập

- Viết code mạng nơ-ron dạng hàm thuần bằng API lập trình hàm của JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Giải thích khác biệt thiết kế then chốt giữa eager mutation của PyTorch và mô hình biên dịch lập trình hàm của JAX
- Áp dụng biên dịch jit và vector hoá vmap để tăng tốc vòng lặp huấn luyện so với Python thuần
- Huấn luyện một mạng đơn giản trong JAX và đối chiếu cách quản lý state tường minh với hướng hướng đối tượng của PyTorch

## Vấn đề

Bạn đã biết cách dựng mạng nơ-ron trong PyTorch. Bạn định nghĩa một `nn.Module`, gọi `.backward()`, cập nhật optimizer. Nó chạy được. Hàng triệu người dùng nó.

Nhưng PyTorch có một ràng buộc nằm sẵn trong DNA của nó: nó truy vết các phép toán theo kiểu eager, từng phép một, trong Python. Mỗi `tensor + tensor` là một lần khởi chạy kernel riêng. Mỗi bước huấn luyện lại diễn giải lại đúng đoạn code Python đó. Cách này ổn cho tới khi bạn cần huấn luyện một mô hình 540 tỷ tham số trên 2.048 TPU. Lúc đó overhead sẽ giết bạn.

Google DeepMind huấn luyện Gemini trên JAX. Anthropic huấn luyện Claude trên JAX. Đây không phải những vận hành nhỏ -- đó là những lần huấn luyện mạng nơ-ron lớn nhất trên Trái Đất. Họ chọn JAX vì nó coi vòng lặp huấn luyện của bạn như một chương trình có thể biên dịch, chứ không phải một chuỗi lệnh gọi Python.

JAX là NumPy với ba siêu năng lực: tự động vi phân, biên dịch just-in-time (biên dịch tức thời) sang XLA, và tự động vector hoá. Bạn viết một hàm xử lý một ví dụ. JAX cho bạn một hàm xử lý cả batch, tính gradient, biên dịch thành mã máy, và chạy trên nhiều thiết bị. Tất cả mà không cần sửa hàm gốc.

## Khái niệm

### Triết lý của JAX

JAX là một framework theo hướng functional programming (lập trình hàm). Không class, không state (trạng thái) khả biến, không phương thức `.backward()`. Thay vào đó:

| PyTorch | JAX |
|---------|-----|
| Class `nn.Module` có state | Hàm thuần: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Thực thi eager | Biên dịch JIT qua XLA |
| `for x in batch:` vòng lặp thủ công | `jax.vmap(f)` tự động vector hoá |
| `DataParallel` / `FSDP` | `jax.pmap(f)` tự động song song |
| `model.parameters()` khả biến | Pytree bất biến của các mảng |

Đây không phải chuyện sở thích phong cách. Đây là ràng buộc của trình biên dịch. Biên dịch JIT đòi hỏi hàm thuần (pure function) -- cùng đầu vào luôn cho cùng đầu ra, không có tác dụng phụ. Chính ràng buộc đó tạo ra khả năng tăng tốc 100 lần.

### jax.numpy: Bề mặt quen thuộc

JAX tái hiện API của NumPy trên các bộ tăng tốc:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Cùng tên hàm. Cùng quy tắc broadcast. Cùng ngữ nghĩa cắt mảng. Nhưng các mảng sống trên GPU/TPU dưới dạng device array (mảng thiết bị), và mọi phép toán đều có thể bị trình biên dịch truy vết.

Một khác biệt quan trọng: mảng JAX là bất biến. Không có `a[0] = 5`. Thay vào đó: `a = a.at[0].set(5)`. Ban đầu sẽ thấy gượng gạo, nhưng một tuần sau nó khớp lại -- tính bất biến là thứ khiến các phép biến đổi như `grad`, `jit`, và `vmap` có thể kết hợp được với nhau.

### jax.grad: Tự động vi phân hàm

PyTorch gắn gradient vào tensor (`.grad`). JAX gắn gradient vào hàm.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` nhận vào một hàm và trả về một hàm mới tính gradient. Không gọi `.backward()`. Không có đồ thị tính toán lưu trên tensor. Gradient chỉ là một hàm khác mà bạn có thể gọi, kết hợp, hoặc biên dịch JIT.

Nó kết hợp được tuỳ ý:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Đạo hàm bậc hai. Đạo hàm bậc ba. Jacobian. Hessian. Tất cả bằng cách kết hợp `grad`. PyTorch cũng làm được điều này (`torch.autograd.functional.hessian`), nhưng nó được gắn thêm vào. Trong JAX, nó là nền tảng.

Ràng buộc: `grad` chỉ hoạt động trên hàm thuần. Không có lệnh print bên trong (chúng chạy trong lúc truy vết, không phải lúc thực thi). Không biến đổi state bên ngoài. Không sinh số ngẫu nhiên mà không quản lý key tường minh.

### jit: Biên dịch sang XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Ở lần gọi đầu tiên, JAX truy vết hàm -- nó ghi lại những phép toán nào diễn ra, mà không thực thi chúng. Sau đó nó chuyển bản truy vết đó cho XLA (Accelerated Linear Algebra), trình biên dịch của Google dành cho TPU và GPU. XLA hợp nhất các phép toán, loại bỏ những lần sao chép bộ nhớ dư thừa, và sinh mã máy đã tối ưu.

Các lần gọi sau bỏ qua Python hoàn toàn. Mã đã biên dịch chạy trên bộ tăng tốc với tốc độ của C++.

Khi JIT có ích:
- Các bước huấn luyện (cùng một phép tính lặp lại hàng nghìn lần)
- Suy luận (cùng mô hình, đầu vào khác nhau)
- Bất kỳ hàm nào được gọi nhiều hơn một lần với đầu vào có hình dạng tương tự

Khi JIT gây hại:
- Hàm có luồng điều khiển Python phụ thuộc vào giá trị (`if x > 0` với x là mảng đang bị truy vết)
- Phép tính chỉ chạy một lần (chi phí biên dịch lớn hơn thời gian chạy)
- Gỡ lỗi (truy vết che mất việc thực thi thật)

Ràng buộc về luồng điều khiển là có thật. `jax.lax.cond` thay cho `if/else`. `jax.lax.scan` thay cho vòng lặp `for`. Đây không phải tuỳ chọn -- đó là cái giá của việc biên dịch.

### vmap: Tự động vector hoá

Bạn viết một hàm xử lý một ví dụ:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` nâng nó lên để xử lý cả batch:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` nghĩa là: không batch theo `params` (dùng chung), batch theo trục 0 của `x`. Không vòng lặp `for` thủ công. Không reshape. Không luồn chiều batch đi khắp nơi. JAX tự tìm ra chiều batch và vector hoá toàn bộ phép tính.

Đây không phải đường ngọt cú pháp. `vmap` sinh ra mã vector hoá đã hợp nhất, chạy nhanh hơn vòng lặp Python từ 10 đến 100 lần. Và nó kết hợp được với `jit` và `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Gradient theo từng ví dụ. Một dòng. Điều này gần như không thể làm trong PyTorch mà không dùng mẹo.

### pmap: Song song dữ liệu trên nhiều thiết bị

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` nhân bản hàm trên tất cả thiết bị có sẵn (GPU/TPU) và chia batch. Bên trong hàm, `jax.lax.pmean` và `jax.lax.psum` đồng bộ gradient giữa các thiết bị.

Google huấn luyện Gemini trên hàng nghìn chip TPU v5e bằng `pmap` (và kế nhiệm của nó là `shard_map`). Mô hình lập trình: viết phiên bản cho một thiết bị, bọc bằng `pmap`, xong.

### Pytree: Cấu trúc dữ liệu phổ quát

JAX vận hành trên "pytree" -- tổ hợp lồng nhau của list, tuple, dict, và mảng. Tham số mô hình của bạn là một pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Mọi phép biến đổi của JAX -- `grad`, `jit`, `vmap` -- đều biết cách duyệt pytree. `jax.tree.map(f, tree)` áp dụng `f` lên mọi lá. Đây là cách optimizer cập nhật tất cả tham số cùng lúc:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Không phương thức `.parameters()`. Không đăng ký tham số. Cấu trúc cây chính là mô hình.

### Hàm thuần so với Hướng đối tượng

PyTorch lưu state bên trong object:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX dùng hàm thuần với state tường minh:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Params được truyền vào. Không gì được lưu lại. Không gì bị biến đổi. Điều này khiến mọi hàm đều kiểm thử được, kết hợp được, và biên dịch được. Nó cũng có nghĩa là bạn tự quản lý params -- hoặc dùng một thư viện như Flax hay Equinox.

### Hệ sinh thái JAX

JAX cho bạn các nguyên thuỷ. Các thư viện cho bạn sự tiện dụng:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Các lớp mạng nơ-ron | `nn.Module` với state tường minh |
| **Equinox** (Patrick Kidger) | Các lớp mạng nơ-ron | Dựa trên pytree, kiểu Python |
| **Optax** (DeepMind) | Optimizer + lịch LR | Biến đổi gradient kết hợp được |
| **Orbax** (Google) | Lưu checkpoint | Lưu/khôi phục pytree |
| **CLU** (Google) | Metrics + logging | Tiện ích cho vòng lặp huấn luyện |

Optax là thư viện optimizer chuẩn. Nó tách việc biến đổi gradient (Adam, SGD, clipping) khỏi bước cập nhật tham số, khiến việc kết hợp trở nên dễ dàng:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Khi nào dùng JAX, khi nào dùng PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| Hỗ trợ TPU | Hạng nhất (Google xây cả hai) | Do cộng đồng duy trì (torch_xla) |
| Hỗ trợ GPU | Tốt (CUDA qua XLA) | Tốt nhất (CUDA gốc) |
| Gỡ lỗi | Khó (truy vết + biên dịch) | Dễ (eager, từng dòng) |
| Hệ sinh thái | Tập trung nghiên cứu (Flax, Equinox) | Khổng lồ (HuggingFace, torchvision, v.v.) |
| Tuyển dụng | Ngách (Google/DeepMind/Anthropic) | Phổ biến (khắp nơi) |
| Huấn luyện quy mô lớn | Vượt trội (XLA, pmap, mesh) | Tốt (FSDP, DeepSpeed) |
| Tốc độ làm nháp | Chậm hơn (overhead của lập trình hàm) | Nhanh hơn (biến đổi rồi chạy) |
| Suy luận production | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Ai dùng | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Câu trả lời thật lòng: hãy dùng PyTorch trừ khi bạn có lý do cụ thể để dùng JAX. Những lý do đó là -- có quyền truy cập TPU, cần gradient theo từng ví dụ, huấn luyện nhiều thiết bị ở quy mô cực lớn, hoặc làm việc tại Google/DeepMind/Anthropic.

### Số ngẫu nhiên trong JAX

JAX không có state ngẫu nhiên toàn cục. Mọi phép toán ngẫu nhiên đều cần một PRNG key tường minh:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Điều này ban đầu gây khó chịu. Nhưng nó bảo đảm khả năng tái lập trên các thiết bị và các lần biên dịch -- một tính chất mà `torch.manual_seed` của PyTorch không bảo đảm được trong môi trường nhiều GPU.

```figure
batchnorm-effect
```

## Tự tay làm

### Bước 1: Cài đặt và dữ liệu

Chúng ta sẽ huấn luyện một MLP 3 lớp trên MNIST bằng JAX và Optax. 784 đầu vào, hai lớp ẩn 256 và 128 nơ-ron, 10 lớp đầu ra.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Bước 2: Khởi tạo tham số

Không class. Chỉ một hàm trả về pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Khởi tạo He, làm thủ công. Ba PRNG key được tách ra từ một seed. Mỗi trọng số là một mảng bất biến trong một dict lồng nhau.

### Bước 3: Lan truyền xuôi

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Hàm thuần. Params vào, dự đoán ra. Không `self`, không state được lưu. `loss_fn` tính cross-entropy từ đầu -- softmax, log, trung bình âm.

### Bước 4: Bước huấn luyện biên dịch JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` trả về cả giá trị loss và gradient trong một lần chạy. Decorator `@jax.jit` biên dịch cả hai hàm sang XLA. Sau lần gọi đầu tiên, mỗi bước huấn luyện chạy mà không chạm tới Python.

### Bước 5: Vòng lặp huấn luyện

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epoch. Độ chính xác test khoảng 97%. Epoch đầu chậm (do biên dịch JIT). Epoch 2-10 nhanh.

Để ý những gì không có: không `.zero_grad()`, không `.backward()`, không `.step()`. Toàn bộ việc cập nhật là một lần gọi hàm đã kết hợp. Gradient được tính, biến đổi bởi Adam, rồi áp dụng lên tham số -- tất cả bên trong `train_step`.

## Áp dụng

### Flax: Chuẩn của Google

Flax là thư viện mạng nơ-ron JAX phổ biến nhất. Nó mang `nn.Module` trở lại, nhưng với quản lý state tường minh:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Cấu trúc giống PyTorch, nhưng `params` tách khỏi mô hình. `model.init()` tạo params. `model.apply(params, x)` chạy lan truyền xuôi. Object mô hình không giữ state.

### Equinox: Lựa chọn kiểu Python

Equinox (của Patrick Kidger) biểu diễn mô hình dưới dạng pytree:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Bản thân mô hình là một pytree. Không cần `.apply()`. Tham số chỉ là các lá của mô hình. Cách này gần với cách JAX suy nghĩ hơn.

### Optax: Optimizer kết hợp được

Optax tách rời việc biến đổi gradient khỏi bước cập nhật:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Clipping gradient, warmup learning rate, weight decay -- tất cả được kết hợp thành một chuỗi biến đổi. Mỗi biến đổi nhận gradient, sửa nó, rồi chuyển cho biến đổi tiếp theo. Không có class optimizer đồ sộ.

## Đóng gói

**Cài đặt:**

```bash
pip install jax jaxlib optax flax
```

Với hỗ trợ GPU:

```bash
pip install jax[cuda12]
```

Với TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Những cái bẫy về hiệu năng:**

- Lần gọi JIT đầu tiên chậm (do biên dịch). Hãy warm up trước khi đo hiệu năng.
- Tránh vòng lặp Python trên mảng JAX bên trong JIT. Dùng `jax.lax.scan` hoặc `jax.lax.fori_loop`.
- `jax.debug.print()` hoạt động bên trong JIT. Còn `print()` thường thì không.
- Profile bằng `jax.profiler` hoặc TensorBoard. Biên dịch XLA có thể che mất các điểm nghẽn.
- JAX mặc định cấp phát trước 75% bộ nhớ GPU. Đặt `XLA_PYTHON_CLIENT_PREALLOCATE=false` để tắt.

**Lưu checkpoint:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Bài học này tạo ra:**
- `outputs/prompt-jax-optimizer.md` -- một prompt để chọn cấu hình optimizer JAX phù hợp
- `outputs/skill-jax-patterns.md` -- một skill bao quát các mẫu lập trình hàm trong JAX

## Bài tập

1. Thêm dropout vào MLP. Trong JAX, dropout cần một PRNG key -- hãy luồn key qua lan truyền xuôi và tách nó cho mỗi lớp dropout. So sánh độ chính xác test khi có và khi không có dropout.

2. Dùng `jax.vmap` để tính gradient theo từng ví dụ cho một batch 32 ảnh MNIST. Tính norm gradient của từng ví dụ. Những ví dụ nào có gradient lớn nhất, và vì sao?

3. Thay hàm lan truyền xuôi thủ công bằng một `mlp_forward(params, x)` tổng quát hoạt động với bất kỳ số lớp nào. Dùng `jax.tree.leaves` để tự xác định độ sâu.

4. Benchmark bước huấn luyện khi có và khi không có `@jax.jit`. Đo thời gian 100 bước cho mỗi trường hợp. Tốc độ tăng bao nhiêu trên phần cứng của bạn? Chi phí biên dịch ở lần gọi đầu tiên là bao nhiêu?

5. Cài đặt clipping gradient bằng cách kết hợp `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Huấn luyện khi có và khi không có clipping. Vẽ norm gradient theo thời gian huấn luyện để thấy tác động.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| XLA | "Thứ làm JAX nhanh" | Accelerated Linear Algebra -- trình biên dịch hợp nhất các phép toán và sinh kernel GPU/TPU tối ưu từ một đồ thị tính toán |
| JIT | "Biên dịch tức thời" | JAX truy vết hàm ở lần gọi đầu, biên dịch sang XLA, rồi chạy bản đã biên dịch ở những lần gọi sau |
| Pure function | "Không có tác dụng phụ" | Hàm mà đầu ra chỉ phụ thuộc vào đầu vào -- không state toàn cục, không biến đổi, không ngẫu nhiên nếu không có key tường minh |
| vmap | "Tự động batch" | Biến đổi một hàm xử lý một ví dụ thành hàm xử lý cả batch, mà không phải viết lại |
| pmap | "Tự động song song" | Nhân bản một hàm trên nhiều thiết bị và chia batch đầu vào |
| Pytree | "Dict lồng nhau của các mảng" | Mọi cấu trúc lồng nhau gồm list, tuple, dict, và mảng mà JAX có thể duyệt và biến đổi |
| Tracing | "Ghi lại phép tính" | JAX thực thi hàm với các giá trị trừu tượng để dựng đồ thị tính toán, mà không tính ra kết quả thật |
| Functional autodiff | "grad của một hàm" | Tính đạo hàm bằng cách biến đổi hàm, chứ không gắn chỗ lưu gradient vào tensor |
| Optax | "Thư viện optimizer của JAX" | Thư viện các phép biến đổi gradient kết hợp được -- Adam, SGD, clipping, lịch học -- nối thành chuỗi |
| Flax | "nn.Module của JAX" | Thư viện mạng nơ-ron của Google cho JAX, thêm lớp trừu tượng trong khi vẫn giữ state tường minh |

## Đọc thêm

- JAX documentation: https://jax.readthedocs.io/ -- tài liệu chính thức, có các hướng dẫn rất tốt về grad, jit, và vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- bài báo gốc giải thích triết lý thiết kế
- Flax documentation: https://flax.readthedocs.io/ -- thư viện mạng nơ-ron của Google cho JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- lựa chọn kiểu Python thay cho Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" -- thư viện optimizer chuẩn
- "You Don't Know JAX" (Colin Raffel, 2020) -- hướng dẫn thực hành về các bẫy và mẫu trong JAX, từ một trong những tác giả của T5
