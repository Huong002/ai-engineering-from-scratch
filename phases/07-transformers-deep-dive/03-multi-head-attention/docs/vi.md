# Multi-Head Attention

> Một head (đầu) attention chỉ học một quan hệ tại một thời điểm. Tám head học tám quan hệ. Head là miễn phí. Cứ lấy thêm nhiều head.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 02 (Self-attention from scratch)
**Time:** ~75 phút

## Vấn đề

Một head self-attention đơn lẻ tính ra một ma trận attention. Ma trận đó nắm bắt một loại quan hệ — thường là loại làm loss nhỏ nhất trên tín hiệu huấn luyện mà bạn đang có. Nếu dữ liệu của bạn có hòa hợp chủ-vị, đồng tham chiếu, diễn ngôn tầm xa và phân đoạn cú pháp trộn lẫn vào nhau, một head đơn lẻ sẽ nhào chúng thành một phân phối soft-max duy nhất và mất đi một nửa tín hiệu.

Cách khắc phục từ bài báo Vaswani 2017: chạy song song nhiều hàm attention, mỗi hàm có projection (phép chiếu) Q, K, V riêng, rồi concatenate (nối các đầu) các output lại. Mỗi head làm việc trong một subspace (không gian con) nhỏ hơn với số chiều `d_model / n_heads`. Tổng số tham số không đổi. Sức biểu diễn tăng lên.

Multi-head attention (attention nhiều đầu) là mặc định mà mọi transformer năm 2026 đều có. Tranh cãi duy nhất là *bao nhiêu* head và liệu key với value có dùng chung projection hay không (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Khái niệm

![Multi-head attention tách, attend, rồi nối các đầu](../assets/multi-head-attention.svg)

**Tách.** Lấy `X` có shape `(N, d_model)`. Chiếu ra Q, K, V, mỗi cái shape `(N, d_model)`. Reshape thành `(N, n_heads, d_head)` với `d_head = d_model / n_heads`. Transpose thành `(n_heads, N, d_head)`.

**Attend song song.** Chạy scaled dot-product attention bên trong từng head. Mỗi head tạo ra `(N, d_head)`. Các head làm việc trên những không gian con khác nhau của embedding và không hề nói chuyện với nhau trong lúc tính attention.

**Nối và chiếu.** Xếp các head trở lại thành `(N, d_model)` rồi nhân với ma trận output đã học `W_o` shape `(d_model, d_model)`. `W_o` là chỗ các head được trộn vào nhau.

**Vì sao nó hiệu quả.** Mỗi head có thể chuyên môn hóa mà không phải giành giật ngân sách biểu diễn với các head khác. Các nghiên cứu probing từ 2019–2024 cho thấy vai trò riêng của từng head: head vị trí, head attend vào token trước đó, head sao chép, head thực thể có tên, head induction (nền tảng của in-context learning).

**Dòng biến thể năm 2026:**

| Biến thể | Số head Q | Số head K/V | Được dùng bởi |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (ví dụ N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | nén về hạng thấp | DeepSeek-V2, V3 |

GQA là mặc định hiện đại vì nó cắt bộ nhớ KV-cache đi `N/G` lần mà vẫn giữ gần như đầy đủ chất lượng. MLA đi xa hơn: nén K/V vào một không gian latent, rồi chiếu ngược lại lúc tính toán — tốn FLOPs, nhưng tiết kiệm bộ nhớ nhiều hơn hẳn.

```figure
multihead-split
```

## Tự tay làm

### Bước 1: tách head từ self-attention một head đã có

Lấy `SelfAttention` từ Bài 02 rồi bọc nó bằng một cặp split/concat. Xem `code/main.py` để có bản cài đặt numpy; logic là:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Một reshape và một transpose. Không vòng lặp. Đây chính xác là thứ PyTorch làm bên dưới `nn.MultiheadAttention`.

### Bước 2: chạy scaled-dot-product attention cho từng head

Mỗi head nhận một lát cắt Q, K, V của riêng nó. Attention trở thành một matmul theo batch:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Trên phần cứng thật, `Qh @ Kh.transpose(...)` là một `bmm`. GPU thấy một matmul theo batch duy nhất với shape `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Thêm head là miễn phí.

### Bước 3: biến thể Grouped-Query Attention

Chỉ projection của key và value thay đổi. Q có `n_heads` nhóm; K và V có `n_kv_heads < n_heads` nhóm và được lặp lại cho khớp:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Khi inference, cách này tiết kiệm bộ nhớ vì chỉ có `n_kv_heads` bản sống trong KV cache, không phải `n_heads`. Llama 3 70B dùng 64 head query với 8 head KV — cache co lại 8×.

### Bước 4: thăm dò xem mỗi head đã học được gì

Chạy MHA trên một câu ngắn với 4 head. Với từng head, in ra ma trận attention `(N, N)`. Bạn sẽ thấy các head khác nhau bắt ra những cấu trúc khác nhau ngay cả khi khởi tạo ngẫu nhiên — phần nào là tín hiệu, phần nào là đối xứng quay trong các không gian con.

## Áp dụng

Trong PyTorch, phiên bản một dòng:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA từ PyTorch 2.5 trở đi:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Bao nhiêu head?** Vài quy tắc ngón tay cái rút ra từ các model production năm 2026:

| Kích thước model | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Nhỏ (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Lớn (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` gần như luôn rơi vào 64 hoặc 128. Nó là đơn vị đo xem một head có thể "nhìn" được bao nhiêu. Xuống dưới 32 là các head bắt đầu đánh nhau với hệ số scaling `sqrt(d_head)`; vượt quá 256 là bạn mất lợi ích "nhiều chuyên gia nhỏ".

## Đóng gói

Xem `outputs/skill-mha-configurator.md`. Skill này gợi ý số head, số head kv và chiến lược projection cho một transformer mới, dựa trên ngân sách tham số, độ dài chuỗi và mục tiêu triển khai.

## Bài tập

1. **Dễ.** Lấy MHA trong `code/main.py` và đổi `n_heads` từ 1 lên 16, giữ `d_model=64`. Vẽ loss của một model tí hon một lớp trên một bài toán copy tổng hợp. Nhiều head hơn thì giúp, bão hòa, hay làm tệ đi?
2. **Trung bình.** Cài đặt MQA (một head KV dùng chung cho mọi head query). Đo xem số tham số giảm bao nhiêu so với MHA đầy đủ. Tính xem kích thước KV-cache co lại bao nhiêu khi inference với N=2048.
3. **Khó.** Cài đặt một phiên bản tí hon của Multi-head Latent Attention: nén K,V thành latent hạng `r`, lưu latent đó trong KV-cache, giải nén lúc tính attention. Với `r` bằng bao nhiêu thì bộ nhớ cache xuống dưới 1/8 của MHA đầy đủ mà chất lượng vẫn nằm trong 1 bit của ppl validation?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Head | "Một mạch attention" | Một phép chiếu Q/K/V với số chiều `d_head = d_model / n_heads`, có ma trận attention riêng. |
| d_head | "Head dimension (số chiều mỗi đầu)" | Độ rộng ẩn của từng head; trong production hầu như luôn là 64 hoặc 128. |
| Split / combine | "Mấy chiêu reshape" | Reshape+transpose `(N, d_model) ↔ (n_heads, N, d_head)` quanh phần attention. |
| W_o | "Phép chiếu output" | Ma trận `(d_model, d_model)` áp sau khi nối các head; là chỗ các head trộn vào nhau. |
| MQA | "Một head KV" | Multi-Query Attention: một phép chiếu K/V dùng chung. KV cache nhỏ nhất, mất một phần chất lượng. |
| GQA | "Mặc định từ Llama 2" | Grouped-Query Attention với `n_kv_heads < n_heads`; lặp lại cho khớp Q. |
| MLA | "Chiêu của DeepSeek" | Multi-head Latent Attention: K,V được nén thành latent hạng thấp, giải nén lúc attend. |
| Induction head | "Mạch đứng sau in-context learning" | Một cặp head phát hiện những lần xuất hiện trước đó và sao chép những gì đi sau chúng. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — đặc tả multi-head nguyên bản.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — bài báo MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — cách chuyển MHA thành GQA sau khi huấn luyện.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA và vì sao nó thắng MHA/GQA về bộ nhớ cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — góc nhìn cơ chế về việc các head thực sự làm gì.

---

[← Bài trước: Self-attention từ đầu](../../02-self-attention-from-scratch/docs/vi.md) · [Bài sau: Positional Encoding →](../../04-positional-encoding/docs/vi.md)
