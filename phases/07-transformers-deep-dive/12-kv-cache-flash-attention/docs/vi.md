# KV Cache, Flash Attention & Tối ưu suy luận

> Huấn luyện thì song song và bị giới hạn bởi FLOP. Suy luận thì tuần tự và bị giới hạn bởi bộ nhớ. Nút thắt khác nhau thì chiêu khác nhau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 02 (Self-attention), Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 7, Bài 07 (GPT)
**Time:** ~75 phút

## Vấn đề

Một decoder tự hồi quy ngây thơ tốn `O(N²)` công để sinh `N` token: ở mỗi bước nó tính lại attention trên toàn bộ tiền tố. Với một câu trả lời 4K token thì đó là 16M phép attention, phần lớn là dư thừa. Mọi hidden state của một token tiền tố là xác định một khi đã tính xong — bạn chỉ cần chạy query của token mới so với key và value đã lưu của mọi thứ phía trước.

Thêm vào đó, bản thân attention di chuyển rất nhiều dữ liệu. Attention chuẩn hiện thực hoá một ma trận điểm số N×N, đầu ra softmax N×d, đầu ra cuối N×d — quá nhiều lượt đọc và ghi vào HBM. Với N≥2K, attention trở nên bị giới hạn bởi bộ nhớ trước khi bị giới hạn bởi FLOP. Các kernel attention cổ điển chỉ dùng được 1/4 đến 1/10 công suất GPU hiện đại.

Hai tối ưu hoá, đều của Dao et al., đã đẩy suy luận tiên tiến từ "chậm" sang "nhanh":

1. **KV cache (bộ nhớ đệm kv).** Lưu các vector K và V của mọi token tiền tố. Attention của mỗi token mới chỉ là một query so với các key đã lưu. Mỗi bước sinh, suy luận giảm từ `O(N²)` xuống `O(N)`.
2. **Flash attention.** Chia ô (tiling) phép tính attention sao cho ma trận N×N đầy đủ không bao giờ chạm HBM. Toàn bộ softmax + matmul diễn ra trong SRAM. Nhanh hơn 2–4× thời gian thực trên A100; 5–10× trên H100 với FP8.

Đến năm 2026 cả hai đều đã phổ cập. Mọi stack suy luận production (vLLM, TensorRT-LLM, SGLang, llama.cpp) đều mặc định dùng chúng. Mọi mô hình tiên tiến đều xuất xưởng với Flash attention bật sẵn.

## Khái niệm

![Sự phình to của kv cache và cách Flash attention chia ô](../assets/kv-cache-flash-attn.svg)

### Phép toán của kv cache

Tính cho mỗi layer decoder, mỗi token, mỗi head:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

Với mô hình 7B có 32 layer, 32 head, d_head=128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Với Llama 3 70B (80 layer, d_head=128, GQA với 8 KV head):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

Chính 10 GB đó là lý do Llama 3 70B ở ngữ cảnh 128K cần gần trọn một A100 40 GB chỉ để chứa kv cache với batch size 1.

**GQA (grouped-query attention) là chiến thắng của kv cache.** MHA với 64 head sẽ là 32 GB. MLA còn nén sâu hơn nữa.

Kéo các chiều rồi xem kích thước cache thay đổi. Tăng độ dài chuỗi hoặc batch lên và thấy nó vượt qua một GPU nhanh đến mức nào:

```figure
kv-cache-sizer
```

### Flash attention — chiêu chia ô

Attention chuẩn:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Ba lượt đi về HBM. Trên H100, băng thông bộ nhớ (memory bandwidth) của HBM là 3 TB/s; SRAM là 30 TB/s. Mỗi lượt chạm HBM chậm hơn 10 lần so với việc giữ mọi thứ trên chip.

Flash attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

Một lượt HBM cho mỗi ô. Tổng dấu chân bộ nhớ giảm từ `O(N²)` xuống `O(N)`. Lượt lan truyền ngược tính lại một số giá trị từ lượt xuôi thay vì lưu chúng — thêm một chiến thắng về bộ nhớ.

**Chiêu số học.** Softmax chạy dồn duy trì `(max, sum)` xuyên các ô để phép chuẩn hoá cuối cùng là chính xác. Đây không phải xấp xỉ — Flash attention cho đầu ra giống hệt từng bit so với attention chuẩn (bỏ qua khác biệt do fp16 không có tính kết hợp).

**Tiến hoá qua các phiên bản:**

| Phiên bản | Năm | Thay đổi chính | Tốc độ trên phần cứng tham chiếu |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Kernel SRAM chia ô | 2× trên A100 |
| Flash 2 | 2023 | Song song hoá tốt hơn, ưu tiên thứ tự nhân quả | 3× trên A100 |
| Flash 3 | 2024 | Bất đồng bộ trên Hopper, FP8 | 1.5–2× trên H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Pipeline 5 tầng trên Blackwell, exp2 bằng phần mềm | Ưu tiên suy luận (ban đầu chỉ có lượt xuôi) |

Flash 4 lúc ra mắt chỉ có lượt xuôi. Huấn luyện vẫn dùng Flash 3. Hỗ trợ GQA và varlen cho Flash 4 còn đang chờ (giữa 2026).

### Giải mã suy đoán (speculative decoding) — chiêu thắng độ trễ còn lại

Mô hình rẻ đề xuất N token. Mô hình lớn xác minh cả N token song song. Nếu bước xác minh chấp nhận k token thì bạn chỉ trả giá một lượt xuôi của mô hình lớn cho k token được sinh. Thường k=3–5 với code và văn xuôi.

Mặc định năm 2026:
- **EAGLE 2 / Medusa.** Các draft head tích hợp dùng chung hidden state của bộ xác minh. Nhanh hơn 2–3× mà không mất chất lượng.
- **Giải mã suy đoán với draft model.** Nhanh hơn 2–4× trên phần cứng phổ thông.
- **Lookahead decoding.** Lặp Jacobi; không cần draft model. Ít phổ biến nhưng miễn phí.

### Continuous batching (ghép lô liên tục)

Suy luận theo lô cổ điển: đợi chuỗi chậm nhất xong rồi mới bắt đầu lô mới. Rất phí GPU khi các câu trả lời ngắn đã xong từ sớm.

Continuous batching (lần đầu có trong Orca, nay có trong vLLM, TensorRT-LLM, SGLang): thay request mới vào lô ngay khi request cũ xong. Thông lượng tăng 5–10× với workload chat thông thường.

### PagedAttention — kv cache như bộ nhớ ảo

Tính năng đinh của vLLM. kv cache được cấp phát theo block 16 token; một page table ánh xạ vị trí logic sang block vật lý. Nhờ đó bạn chia sẻ được KV giữa các mẫu song song (beam search, parallel sampling), hoán đổi nóng tiền tố cho prompt caching, và chống phân mảnh bộ nhớ. Thông lượng cải thiện 4× so với cấp phát liền mạch ngây thơ.

```figure
flash-attention-memory
```

## Tự tay làm

Xem `code/main.py`. Chúng ta cài đặt:

1. Một decoder tăng dần `O(N²)` ngây thơ.
2. Một decoder có kv cache `O(N)`.
3. Một softmax chia ô mô phỏng thuật toán running-max của Flash attention.

### Bước 1: kv cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Đơn giản: cứ nối thêm vector K, V của từng token vào các list theo layer, theo head.

### Bước 2: softmax chia ô

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Đầu ra giống hệt từng bit so với `softmax(qK) V` tính một lần, nhưng ở bất kỳ thời điểm nào working set cũng chỉ là một block `tile × d_head`, không phải toàn bộ `N × d_head`.

### Bước 3: so sánh giải mã (decode) ngây thơ và có cache trên 100 token

Đếm số phép attention. Ngây thơ: `O(N²)` = 5050. Có cache: `O(N)` = 100. Code in ra cả hai.

## Áp dụng

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM trong production:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Prefix caching giữa các request là một chiến thắng lớn của năm 2026 — cùng một system prompt, vài ví dụ few-shot, hay một tài liệu ngữ cảnh dài đều tái dùng KV qua các lần gọi. Với workload agent có prompt tool lặp đi lặp lại, prefix caching thường đem lại thông lượng gấp 5×.

## Đóng gói

Xem `outputs/skill-inference-optimizer.md`. Skill này chọn cách cài đặt attention, chiến lược kv cache, lượng tử hoá, và giải mã suy đoán cho một triển khai suy luận mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác nhận decoder ngây thơ và decoder có cache cho cùng đầu ra; để ý chênh lệch số phép tính.
2. **Trung bình.** Cài đặt prefix caching: cho một prompt P và vài completion, chạy một lượt xuôi trên P để nạp đầy kv cache, rồi rẽ nhánh cho từng completion. Đo mức tăng tốc so với việc mã hoá lại P cho từng completion.
3. **Khó.** Cài đặt một PagedAttention đồ chơi: kv cache theo block cố định 16 token kèm free-list. Khi một chuỗi kết thúc, trả block của nó về pool. Mô phỏng 1.000 chat completion với độ dài khác nhau. So sánh phân mảnh bộ nhớ với cấp phát liền mạch.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| KV cache | "Chiêu giúp giải mã nhanh" | K và V đã lưu của mọi token tiền tố; query mới chú ý tới chúng thay vì tính lại. |
| HBM | "Bộ nhớ chính của GPU" | High Bandwidth Memory; 80 GB trên H100, 192 GB trên B200. Băng thông ~3 TB/s. |
| SRAM | "Bộ nhớ trên chip" | Bộ nhớ nhanh riêng của từng SM, ~256 KB mỗi SM trên H100. Băng thông ~30 TB/s. |
| Flash Attention | "Kernel attention chia ô" | Tính attention mà không hiện thực hoá ma trận N×N trong HBM. |
| Continuous batching | "Batch không phải chờ" | Đưa chuỗi đã xong ra, đưa chuỗi mới vào, không cần xả cả batch. |
| PagedAttention | "Tính năng đinh của vLLM" | kv cache cấp phát theo block cố định kèm page table; xoá bỏ phân mảnh. |
| Prefix caching | "Tái dùng prompt dài" | Cache KV cho tiền tố dùng chung giữa các request; cắt chi phí lớn cho agent. |
| Speculative decoding | "Nháp + xác minh" | Mô hình nháp rẻ đề xuất token; mô hình lớn xác minh k token trong một lượt. |

## Đọc thêm

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — pipeline 5 tầng trên Blackwell và chiêu exp2 bằng phần mềm; đọc README của repo để biết các lưu ý khi ra mắt chỉ có lượt xuôi mà bài này nhắc tới.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — bài báo vLLM.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — giải mã suy đoán.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — bài báo EAGLE-1/2 về cách tiếp cận draft tích hợp mà bài này dẫn.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — cách tiếp cận Medusa được nhắc cùng EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — bài đi sâu kinh điển về thiết kế block 16 token và page table.

---

[← Bài trước: Mixture of Experts (MoE)](../../11-mixture-of-experts/docs/vi.md) · [Bài sau: Quy luật tỉ lệ (Scaling Laws) →](../../13-scaling-laws/docs/vi.md)
