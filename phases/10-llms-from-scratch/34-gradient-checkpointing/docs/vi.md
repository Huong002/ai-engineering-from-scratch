# Checkpoint gradient và tính lại hoạt hoá

> Lượt lan truyền ngược (backward pass) giữ lại mọi hoạt hoá trung gian. Với 70B tham số và ngữ cảnh 128K, đó là 3 TB hoạt hoá cho mỗi rank. Checkpointing đánh đổi (trade-off) FLOPs lấy bộ nhớ: tính lại thay vì lưu. Câu hỏi là nên bỏ những đoạn nào, và câu trả lời không phải "bỏ tất cả".

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Giai đoạn 10, Bài 04 (huấn luyện trước mini-GPT), Giai đoạn 10, Bài 05 (mở rộng quy mô & phân tán)
**Time:** ~70 phút

## Vấn đề

Huấn luyện một transformer lưu, cho mỗi lớp, đầu vào của mọi phép toán được lấy đạo hàm trong lượt lan truyền ngược: đầu vào attention, các phép chiếu Q/K/V, đầu ra softmax, đầu vào FFN, đầu ra của norm, và luồng residual. Với một lớp có hidden size `d`, độ dài chuỗi `L`, batch `B`, con số này vào cỡ `12 * B * L * d` số thực mỗi lớp.

Với `d=8192, L=8192, B=1`, đó là 800 MB/lớp ở BF16. Một mô hình 64 lớp là 51 GB hoạt hoá — và đó là trước khi nhân với kích thước microbatch, trước khi cộng thêm các giá trị trung gian attention-softmax (`L^2` mỗi head), và trước khi tính đến các bản sao cục bộ của tensor song song.

Hai phía của hoá đơn: trọng số BF16 cộng trạng thái optimizer có thể vừa trong 80GB, nhưng hoạt hoá đẩy bạn vượt qua. Gradient checkpointing (checkpoint gradient), hay còn gọi là tính lại hoạt hoá, là cách sửa tiêu chuẩn. Bỏ phần lớn hoạt hoá; làm lại lượt forward trong lượt backward để lấy chúng về. Cái giá: thêm FLOPs. Lợi ích: memory footprint (mức chiếm bộ nhớ) giảm theo tỉ lệ giữa số đoạn checkpoint và tổng số lớp.

Làm ngây thơ, checkpointing tốn thêm khoảng 33% FLOPs forward mỗi bước. Làm khéo — checkpoint chọn lọc theo "smart selection" của Korthikanti và cộng sự — bạn tiết kiệm 5 lần bộ nhớ với chi phí dưới 5% FLOPs. Và với matmul FP8, offload FSDP, và MoE expert-parallel thì điều này thực sự quan trọng: bạn không đủ sức chịu cả bộ nhớ lẫn phần tính toán lãng phí.

## Khái niệm

### Lượt lan truyền ngược thực sự cần gì

`output = layer(input)`. Lượt lan truyền ngược muốn `grad_input` và `grad_params`. Để tính chúng nó cần:

- `input` (để tính `grad_params = input.T @ grad_output` cho lớp tuyến tính)
- một số giá trị trung gian đạo hàm của hoạt hoá (đạo hàm của ReLU/GELU/softmax phụ thuộc vào giá trị hoạt hoá)

Lượt forward lưu những thứ này tự động trong đồ thị autograd. Mỗi `tensor.retain_grad()` và mỗi phép toán cần đầu vào của nó đều giữ một tham chiếu.

### Checkpoint toàn phần kiểu ngây thơ

Chia mạng thành `N` đoạn. Trong lượt forward, chỉ lưu *đầu vào* của mỗi đoạn. Khi lượt backward cần giá trị trung gian, chạy lại lượt forward của đoạn đó để vật chất hoá chúng, rồi lấy đạo hàm.

Ví dụ: transformer 32 lớp chia thành 32 đoạn, mỗi đoạn 1 lớp.

- Bộ nhớ: 32 đầu vào lớp (nhỏ) so với 32 * (khối lượng hoạt hoá mỗi lớp) (khổng lồ).
- Tính toán thêm: 1 lượt forward thêm mỗi đoạn, tức tổng cộng thêm ~33% FLOPs forward (vì backward gấp 2 lần forward, cả bước thành 1 + 1 + 2 = 4 đơn vị thay vì 1 + 2 = 3).

Đây là công thức gốc của Chen và cộng sự 2016: một checkpoint mỗi `sqrt(L)` lớp để cân bằng bộ nhớ và tính toán. Với L=64, tức 8 checkpoint.

### Checkpoint chọn lọc (Korthikanti 2022)

Không phải mọi hoạt hoá đều tốn như nhau. Đầu ra softmax của attention là `B*L*L*heads` và tăng *bậc hai* theo độ dài chuỗi. Hoạt hoá ẩn của FFN là `B*L*4d` và tăng tuyến tính. Với chuỗi dài, softmax chiếm ưu thế.

Checkpoint chọn lọc giữ những hoạt hoá rẻ để lưu (các phép chiếu tuyến tính, residual) và chỉ tính lại những cái đắt (attention). Bạn trả rất ít FLOPs để tính lại nhưng tiết kiệm được bộ nhớ O(L^2).

Megatron-Core cài đặt điều này thành tính lại hoạt hoá "selective". Được dùng trong hầu hết các đợt huấn luyện frontier từ 2024 trở đi.

### Offload

Cách khác thay cho tính lại: chuyển hoạt hoá sang RAM CPU giữa forward và backward. Cần băng thông PCIe; có lợi khi băng thông nhàn rỗi vượt chi phí vật chất hoá lại. Chiến lược hỗn hợp là phổ biến: checkpoint một số lớp, offload số khác.

FSDP2 cung cấp offload như một tuỳ chọn hạng nhất. Offload toả sáng khi GPU bị nghẽn bộ nhớ nhưng đường truyền CPU-GPU còn dư địa.

### Mô hình chi phí tính lại

FLOPs mỗi bước với checkpoint ngây thơ mỗi `k` lớp trong `L` lớp:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Với checkpoint chọn lọc bạn chỉ tính lại kernel attention, không phải cả lớp:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Mô hình tiết kiệm bộ nhớ

Khối lượng hoạt hoá mỗi lớp: `A`. Với `L` lớp, tổng bộ nhớ hoạt hoá: `L * A`.

Checkpoint toàn phần (kích thước đoạn 1): chỉ lưu `L * input_volume` (~`L * 1/10 A` với một transformer chuẩn). Tiết kiệm ~`9 * L * A * 1/10`.

Checkpoint mỗi `k` lớp: lưu `L/k * A` cộng thêm phần của `k-1` lớp trong đoạn đang hoạt động.

Tại `k = sqrt(L)`, bộ nhớ và chi phí tính lại đều tăng theo `sqrt(L)` — đánh đổi tối ưu cho các lớp có chi phí đồng đều.

### Khi nào không nên checkpoint

- Những lớp trong cùng của một stage pipeline đang chạy dở. Dù sao chúng cũng phải hoàn thành.
- Lớp đầu và lớp cuối nếu chúng chiếm phần lớn tính toán của stage (hiếm với transformer).
- Kernel attention đã dùng FlashAttention — Flash vốn đã tính lại softmax rất nhanh, nên checkpoint thêm ở mức lớp chẳng thêm được bao nhiêu.

### Các mẫu cài đặt

1. **Bọc hàm:** bọc một đoạn trong `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch chỉ lưu `input`, tính lại mọi thứ khác khi backward.

2. **Dùng decorator:** đánh dấu các lớp là có thể checkpoint; trainer quyết định lúc cấu hình đoạn nào được bọc.

3. **Tính lại thủ công tường minh:** tự viết lượt backward, gọi một `recompute_forward` tuỳ biến sao lại lượt forward với đầu vào đã lưu.

Cả ba cho cùng một kết quả chức năng. Bọc hàm là cách viết chuẩn.

### Tương tác với TP / PP / FP8

- **Tensor song song:** đầu vào checkpoint phải được gather hoặc scatter lại khi tính lại; xử lý chi phí truyền thông.
- **Pipeline song song:** mẫu điển hình là checkpoint lượt forward của từng stage pipeline để các microbatch theo thứ tự ngược có thể dùng lại bộ nhớ hoạt hoá.
- **Tính lại FP8:** lịch sử amax cập nhật trong lúc tính lại phải khớp với lượt forward gốc, nếu không scale FP8 sẽ trôi. Hầu hết framework chụp lại scale.

```figure
activation-recompute
```

## Tự tay làm

### Bước 1: Một mô hình đồ chơi với các đoạn

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Bước 2: Lượt backward ngây thơ cần mọi hoạt hoá

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Bước 3: Bộ nhớ khi checkpoint mỗi k lớp

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Bước 4: Mô hình chi phí

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Bước 5: Bộ ước lượng bộ nhớ

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Bước 6: Kích thước đoạn tối ưu

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Bước 7: Quyết định checkpoint chọn lọc

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Áp dụng

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — wrapper chuẩn trong PyTorch. Bọc một hàm; chỉ lưu đầu vào, tính lại khi backward.
- **Megatron-Core activation recomputation**: hỗ trợ chế độ `selective`, `full`, và `block`. Chuẩn trong huấn luyện frontier từ 2024 trở đi.
- **FSDP2 offload**: `module.to_empty(device="cpu")` với `offload_policy` trong FSDP2 chia nhỏ hoạt hoá sang CPU thay vì tính lại.
- **DeepSpeed ZeRO-Offload**: offload CPU cho trạng thái optimizer và hoạt hoá, bổ trợ cho checkpointing.

## Đóng gói

Bài học này tạo ra `outputs/prompt-activation-recompute-policy.md` — một prompt nhận cấu hình mô hình của bạn (layers, hidden, seq, batch) và bộ nhớ GPU khả dụng rồi xuất ra chính sách tính lại theo từng lớp (none / selective / full / offload).

## Bài tập

1. Kiểm tra tính đúng đắn. Chạy `model_forward` + `model_backward` (đủ hoạt hoá) so với `model_forward_checkpointed` + `model_backward_checkpointed` (theo đoạn). Gradient của tham số phải trùng khớp đến độ chính xác máy.

2. Quét kích thước đoạn `k` từ 1 đến `L`. Vẽ chi phí FLOPs và bộ nhớ. Tìm điểm gãy của đường cong.

3. Cài đặt checkpoint chọn lọc: lưu đầu vào của module attention nhưng không lưu các giá trị trung gian của nó. Đo chi phí FLOPs so với checkpoint toàn lớp cho mô hình 32 lớp ở seq=8192.

4. Thêm offload. Lưu đầu vào đoạn vào một "CPU buffer" mô phỏng (một list riêng). Đo "băng thông PCIe" theo byte/thời gian và tìm điểm hoà vốn giữa offload và tính lại.

5. Benchmark một transformer PyTorch thật có và không có `torch.utils.checkpoint`. Đo bộ nhớ (qua `torch.cuda.max_memory_allocated`) và thời gian mỗi bước.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Gradient checkpointing | "Tiết kiệm bộ nhớ bằng cách làm lại forward" | Chỉ lưu đầu vào của đoạn; tính lại giá trị trung gian khi backward để lấy các tensor cần cho gradient |
| Activation recomputation | "Giống hệt checkpointing" | Tên gọi mang màu sắc HPC của cùng một kỹ thuật |
| Segment size (k) | "Bao nhiêu lớp mỗi checkpoint" | Số lớp có giá trị trung gian bị bỏ đi và được vật chất hoá lại cùng nhau |
| Selective checkpointing | "Mẹo của Korthikanti" | Chỉ tính lại những hoạt hoá đắt để lưu (softmax của attention); giữ những cái rẻ |
| Full checkpointing | "Bản ngây thơ" | Tính lại giá trị trung gian của mọi lớp trong mọi đoạn |
| Block checkpointing | "Hạt thô" | Checkpoint cả khối transformer; độ hạt lớn nhất |
| FLOP overhead | "Thuế tính toán" | FLOPs thêm mỗi bước = (FLOPs tính lại) / (FLOPs fwd + bwd); 33% ngây thơ, 5% chọn lọc |
| Activation offload | "Chuyển sang CPU" | Đưa hoạt hoá sang RAM CPU trong khoảng forward->backward; thay thế cho tính lại |
| sqrt-L rule | "Điểm tối ưu cổ điển" | Với các lớp chi phí đồng đều, khoảng cách checkpoint tối ưu là sqrt(L) lớp |
| Attention-softmax volume | "Vấn đề O(L^2)" | L^2 * heads * batch số thực; chiếm ưu thế bộ nhớ hoạt hoá khi ngữ cảnh dài |

## Đọc thêm

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- bài báo gốc đã hình thức hoá checkpoint gradient
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- tính lại hoạt hoá chọn lọc và phân tích chi phí chính thức
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- hướng tiếp cận bộ nhớ hằng qua vật chất hoá lại theo chiều ngược
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- offload hoạt hoá ở quy mô lớn
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- API tiêu chuẩn
- [Megatron Bridge activation recomputation documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html) -- chế độ selective, full, và block

---

[← Bài trước: Giải mã suy đoán và EAGLE](../../25-speculative-decoding/docs/vi.md)
