# Cơ chế chú ý (attention) — Bước đột phá

> Decoder thôi nheo mắt nhìn vào một bản tóm tắt bị nén và bắt đầu nhìn vào toàn bộ nguồn. Mọi thứ sau đây là attention cộng với công việc kỹ thuật.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 09 (Mô hình chuỗi sang chuỗi)
**Time:** ~45 phút

## Vấn đề

Bài 09 kết thúc bằng một thất bại đã được đo đạc. Một encoder-decoder GRU huấn luyện trên tác vụ sao chép đồ chơi đi từ độ chính xác 89% ở độ dài 5 xuống gần mức ngẫu nhiên ở độ dài 80. Lý do là cấu trúc, không phải lỗi huấn luyện: mọi mẩu thông tin encoder thu được đều phải nằm vừa trong một trạng thái ẩn có kích thước cố định, và decoder không bao giờ thấy gì khác.

Bahdanau, Cho và Bengio công bố một bản sửa ba dòng vào năm 2014. Thay vì chỉ đưa cho decoder trạng thái encoder cuối cùng, hãy giữ lại mọi trạng thái encoder. Ở mỗi bước decoder, tính một trung bình có trọng số của các trạng thái encoder, trong đó trọng số nói "decoder cần nhìn vào vị trí encoder `i` bao nhiêu ngay lúc này?" Trung bình có trọng số đó là ngữ cảnh, và nó thay đổi ở mỗi bước decoder.

Đó là toàn bộ ý tưởng. Transformer mở rộng nó. Self-attention áp dụng nó cho một chuỗi duy nhất. Multi-head attention chạy nó song song. Nhưng phiên bản 2014 đã phá được nút thắt, và khi bạn đã có nó, bước chuyển sang transformer là công việc kỹ thuật, không phải khái niệm.

## Khái niệm

![Attention Bahdanau: decoder truy vấn mọi trạng thái encoder](../assets/attention.svg)

Tại mỗi bước decoder `t`:

1. Dùng trạng thái ẩn trước đó của decoder `s_{t-1}` làm **query** (truy vấn).
2. Chấm điểm nó với mọi trạng thái ẩn của encoder `h_1, ..., h_T`. Một số vô hướng cho mỗi vị trí encoder.
3. Đưa các điểm qua softmax để được trọng số chú ý `α_{t,1}, ..., α_{t,T}` có tổng bằng 1.
4. Vector ngữ cảnh (context vector) `c_t = Σ α_{t,i} * h_i`. Trung bình có trọng số của các trạng thái encoder.
5. Decoder nhận `c_t` cộng với token đầu ra trước đó, rồi sinh ra token tiếp theo.

Trung bình có trọng số chính là điểm mấu chốt. Khi decoder cần dịch "Je" thành "I", nó cho trọng số cao cho trạng thái encoder ứng với "Je" và thấp cho các trạng thái khác. Khi cần "not", nó cho "pas" trọng số cao. Vector ngữ cảnh định hình lại từng bước.

## Kích thước tensor (thứ khiến ai cũng vấp)

Đây là chỗ mà mọi bản cài đặt attention đều sai trong lần đầu. Đọc chậm thôi.

| Thứ | Shape | Ghi chú |
|-------|-------|-------|
| Trạng thái ẩn của encoder `H` | `(T_enc, d_h)` | Nếu là BiLSTM, `d_h = 2 * d_hidden` |
| Trạng thái ẩn của decoder `s_{t-1}` | `(d_s,)` | Một vector |
| Điểm chú ý (attention score) `e_{t,i}` | số vô hướng | Một điểm cho mỗi vị trí encoder |
| Trọng số chú ý `α_{t,i}` | số vô hướng | Sau softmax trên toàn bộ `i` |
| Vector ngữ cảnh `c_t` | `(d_h,)` | Cùng shape với một trạng thái encoder |

**Điểm chú ý Bahdanau (additive).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` có shape `(d_s,)`, `h_i` có shape `(d_h,)`.
- `W_a` có shape `(d_attn, d_s)`. `U_a` có shape `(d_attn, d_h)`.
- Tổng của chúng bên trong tanh có shape `(d_attn,)`.
- `v_α` có shape `(d_attn,)`. Tích trong với `v_α` thu gọn thành một số vô hướng. **Đây là việc `v_α` làm.** Không có gì huyền bí. Nó là phép chiếu biến một vector ở chiều attention thành một điểm số vô hướng.

**Điểm chú ý Luong (multiplicative).** Ba biến thể:

- `dot`: `e_{t,i} = s_t^T * h_i`. Yêu cầu `d_s == d_h`. Ràng buộc cứng. Bỏ qua nếu encoder của bạn là hai chiều.
- `general`: `e_{t,i} = s_t^T * W * h_i` với `W` có shape `(d_s, d_h)`. Loại bỏ ràng buộc cùng chiều.
- `concat`: về cơ bản là dạng Bahdanau. Ít khi dùng vì hai cách trên rẻ hơn.

**Một cái bẫy Bahdanau / Luong đáng gọi tên.** Bahdanau dùng `s_{t-1}` (trạng thái decoder *trước* khi sinh ra từ hiện tại). Luong dùng `s_t` (trạng thái *sau*). Nhầm lẫn hai cái này tạo ra gradient sai một cách tinh vi, cực khó debug. Chọn một bài báo và bám theo quy ước của nó.

```figure
attention-heatmap
```

## Tự tay làm

### Bước 1: attention cộng tính (Bahdanau)

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Đối chiếu shape của bạn với bảng ở trên. `encoder_states` có shape `(T_enc, d_h)`. `projected_enc` có shape `(T_enc, d_attn)`. `projected_dec` có shape `(d_attn,)` và được broadcast. `combined` có shape `(T_enc, d_attn)`. `scores` có shape `(T_enc,)`. `weights` có shape `(T_enc,)`. `context` có shape `(d_h,)`. Xong, dùng được.

### Bước 2: Luong dot và general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Mỗi cái ba dòng. Đó là lý do bài báo của Luong được chấp nhận. Cùng độ chính xác trên hầu hết tác vụ, ít code hơn nhiều.

### Bước 3: một ví dụ số đã tính sẵn

Cho ba trạng thái encoder (đại khái là "cat", "sat", "mat") và một trạng thái decoder căn chỉnh (alignment) nhiều nhất với cái đầu tiên, phân bố attention dồn về vị trí 0. Nếu trạng thái decoder dịch sang căn chỉnh với cái cuối, attention chuyển sang vị trí 2. Vector ngữ cảnh bám theo.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Hàng đầu thắng. Sau đó đưa trạng thái decoder gần hơn với trạng thái encoder thứ ba và xem các trọng số dịch chuyển. Chỉ có thế. Attention là căn chỉnh tường minh.

### Bước 4: vì sao đây là cầu nối sang transformer

Dịch ngôn ngữ ở trên sang Q/K/V:

- **Query** (truy vấn) = trạng thái decoder `s_{t-1}`
- **Key** (khoá) = các trạng thái encoder (thứ ta chấm điểm)
- **Value** (giá trị) = các trạng thái encoder (thứ ta gán trọng số rồi cộng lại)

Trong attention cổ điển, key và value là cùng một thứ. Self-attention tách chúng ra: bạn có thể truy vấn một chuỗi với chính nó, với các phép chiếu học được khác nhau cho K và V. Multi-head attention chạy việc đó song song với các phép chiếu học được khác nhau. Transformer xếp chồng toàn bộ giai đoạn này nhiều lần và bỏ RNN.

Toán học giống nhau. Các shape giống nhau. Bước nhảy sư phạm từ attention Bahdanau sang scaled dot-product attention hầu như chỉ là ký hiệu.

## Áp dụng

PyTorch và TensorFlow cung cấp attention trực tiếp.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

Đó là một lớp attention của transformer. Batch query 5 vị trí, batch key/value 10 vị trí, mỗi cái 128 chiều, 8 head. `output` là các query mới đã được bổ sung ngữ cảnh. `weights` là ma trận căn chỉnh 5x10 mà bạn có thể trực quan hoá.

### Khi nào attention cổ điển vẫn còn quan trọng

- Sư phạm. Phiên bản một head, một lớp, dựa trên RNN khiến mọi khái niệm hiện ra rõ ràng.
- Các tác vụ chuỗi trên thiết bị, nơi transformer không vừa.
- Mọi bài báo từ 2014-2017. Bạn sẽ đọc sai nếu không biết quy ước của Bahdanau.
- Phân tích căn chỉnh chi tiết trong MT. Trọng số attention thô là một công cụ diễn giải, kể cả trên mô hình transformer, và đọc chúng đòi hỏi phải biết chúng là gì.

### Cái bẫy lấy trọng số chú ý làm lời giải thích

Trọng số attention trông có vẻ diễn giải được. Chúng là các trọng số có tổng bằng một trên các vị trí; bạn có thể vẽ chúng; cao nghĩa là "đã nhìn vào chỗ này". Người review rất thích chúng.

Chúng không diễn giải được như vẻ ngoài. Jain và Wallace (2019) chỉ ra rằng phân bố attention có thể bị hoán vị và thay bằng các phương án tuỳ ý mà không làm thay đổi dự đoán của mô hình trên một số tác vụ. Đừng bao giờ báo cáo trọng số attention như bằng chứng của lập luận nếu không có kiểm tra ablation hoặc counterfactual.

## Đóng gói

Lưu thành `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Bài tập

1. **Dễ.** Cài đặt mặt nạ (masked attention) cho `softmax` để các token padding trong encoder nhận trọng số attention bằng không. Kiểm thử trên một batch có các chuỗi độ dài khác nhau.
2. **Trung bình.** Thêm multi-head attention vào dạng `general` của Luong. Chia `d_h` thành `n_heads` nhóm, chạy attention cho từng head, rồi nối lại. Kiểm tra rằng trường hợp một head khớp với bản cài đặt trước đó.
3. **Khó.** Huấn luyện một encoder-decoder GRU với attention Bahdanau trên tác vụ sao chép đồ chơi ở bài 09. Vẽ đồ thị độ chính xác theo độ dài chuỗi. So sánh với baseline không có attention. Bạn sẽ thấy khoảng cách rộng ra khi độ dài tăng, xác nhận attention nâng được nút thắt.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Attention | Nhìn vào mọi thứ | Trung bình có trọng số của một chuỗi giá trị, trọng số tính từ độ tương đồng query-key. |
| Query, Key, Value | QKV | Ba phép chiếu: Q hỏi, K là thứ để so khớp, V là thứ để trả về. |
| Additive attention | Bahdanau | Điểm bằng feed-forward: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | Điểm là `q^T k` hoặc `q^T W k`. Rẻ hơn, cùng độ chính xác trên hầu hết tác vụ. |
| Alignment matrix | Bức hình đẹp | Trọng số attention dưới dạng lưới `(T_dec, T_enc)`. Đọc nó để xem mô hình đã chú ý tới đâu. |

## Đọc thêm

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — bài báo.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — ba biến thể điểm số và so sánh của chúng.
- [Jain và Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — lưu ý về khả năng diễn giải.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — hướng dẫn chạy được với PyTorch.

---

[← Bài trước: Mô hình sequence-to-sequence (chuỗi sang chuỗi)](../../09-sequence-to-sequence/docs/vi.md) · [Bài sau: Dịch máy →](../../11-machine-translation/docs/vi.md)
