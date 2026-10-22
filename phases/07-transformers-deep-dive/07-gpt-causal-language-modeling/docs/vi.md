# GPT — Mô hình ngôn ngữ nhân quả

> BERT thấy cả hai phía. GPT chỉ thấy quá khứ. Mặt nạ tam giác là dòng code có sức nặng lớn nhất trong AI hiện đại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 02 (Self-attention), Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 7, Bài 06 (BERT)
**Time:** ~75 phút

## Vấn đề

Một mô hình ngôn ngữ chỉ trả lời một câu hỏi: cho trước `t-1` token đầu, phân phối xác suất trên token `t` là gì? Huấn luyện trên tín hiệu đó — next-token prediction (dự đoán token kế tiếp) — và bạn có một mô hình sinh được văn bản tuỳ ý, mỗi lần một token.

Để huấn luyện nó end-to-end trên cả một chuỗi một cách song song, bạn cần dự đoán ở mỗi vị trí chỉ phụ thuộc vào các vị trí đứng trước. Nếu không, mô hình sẽ gian lận một cách quá dễ dàng bằng cách nhìn thẳng vào đáp án.

Causal mask (mặt nạ nhân quả) làm việc này. Nó chỉ là một ma trận tam giác trên gồm các giá trị `-inf`, được cộng vào điểm attention trước softmax. Sau softmax, các vị trí đó thành 0. Mỗi vị trí chỉ có thể attend tới chính nó và các vị trí trước đó. Và vì bạn áp nó một lần cho cả chuỗi, bạn có N dự đoán token kế tiếp song song trong một lượt forward.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2025), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — tất cả đều là transformer nhân quả decoder-only (chỉ decoder) với cùng một vòng lặp lõi. Điều phân biệt chúng là chất lượng dữ liệu, quy mô cùng các tinh chỉnh kiến trúc, và post-training (SFT, RLHF, DPO và những phương pháp kế tiếp).

## Khái niệm

![Causal mask tạo ra một ma trận attention hình tam giác](../assets/causal-attention.svg)

### Mặt nạ

Cho một chuỗi độ dài `N`, dựng một ma trận `N × N`:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Cộng `M` vào điểm attention thô trước softmax. `exp(-inf) = 0`, nên các vị trí bị che đóng góp trọng số bằng không. Mỗi hàng của ma trận attention là một phân phối xác suất chỉ trên các vị trí trước đó.

Chi phí triển khai: một lời gọi `torch.tril()`. Thời gian tính: nano giây. Ảnh hưởng tới cả ngành: mọi thứ.

### Hình tam giác đến từ đâu

Người ta thường trình bày mặt nạ như một miếng vá gắn thêm vào attention. Chạy phép suy dẫn theo chiều ngược lại thì nó không còn bí ẩn nữa: attention là lần tinh chỉnh thứ ba của một trung bình tiền tố, và hình tam giác chính là biên của vòng lặp của trung bình đó, viết ra dưới dạng ma trận.

**Giai đoạn 1 — trung bình tiền tố.** Cách tóm tắt nhân quả ngốc nghếch nhất cho một chuỗi: vị trí `i` trở thành trung bình của các vị trí `0…i`. Viết như một vòng lặp, đó là `out[i] = X[:i+1].mean(0)`. Cùng phép tính đó chỉ là một phép nhân ma trận. Lấy một ma trận tam giác dưới toàn số một, chia mỗi hàng cho số phần tử của nó, rồi nhân:

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

Hàng `i` của `A` là `[1/(i+1), …, 1/(i+1), 0, …, 0]`. Các số không nằm phía trên đường chéo chính là tính nhân quả. Không có gì thuộc về tương lai bị che đi; tương lai vốn dĩ không nằm trong tổng.

**Giai đoạn 2 — trọng số học được.** Một trung bình đều coi mọi token quá khứ đều quan trọng như nhau. Thay các số một bằng một ma trận điểm học được `S`. Giờ các hàng không còn tự động cộng lại thành một, nên hãy chuẩn hoá từng hàng bằng softmax thay vì chia cho số phần tử. Softmax không bao giờ cho ra đúng số không, điều đó phá vỡ tính nhân quả — trừ khi điểm của tương lai được đưa vào dưới dạng `-inf`, vì `exp(-inf) = 0`:

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

Vẫn hình tam giác đó, vẫn ma trận hàng-ngẫu-nhiên đó, vẫn một phép nhân ma trận đó. Mặt nạ `-inf` không phải một cơ chế mới. Nó là các phần tử bằng không của giai đoạn 1, được dịch sang miền đầu vào của softmax.

**Giai đoạn 3 — trọng số phụ thuộc nội dung.** Ở giai đoạn 2, `S` cố định sau khi huấn luyện: vị trí 7 luôn đánh trọng số cho vị trí 3 như nhau, bất kể các token nói gì. Hãy để điểm số phụ thuộc vào chính các token: `S = Q @ K.T / sqrt(d_k)`. Không gì khác thay đổi. Mặt nạ, softmax, phép nhân ma trận — y hệt.

Ba giai đoạn, một bất biến: một ma trận tam giác dưới hàng-ngẫu-nhiên nhân với chuỗi. Trung bình đều, trọng số tĩnh học được, trọng số phụ thuộc nội dung. Mặt nạ chưa bao giờ được thêm vào attention. Nó sống sót từ phép trung bình.

```figure
mask-derivation
```

### Huấn luyện song song, suy luận tuần tự

Huấn luyện: forward cả chuỗi `(N, d_model)` một lần, tính N mất mát cross-entropy (mỗi vị trí một cái), cộng lại, lan truyền ngược. Song song dọc theo chuỗi. Đây là lý do việc huấn luyện GPT mở rộng được — bạn xử lý 1M token trong một batch chỉ bằng một lượt GPU.

Suy luận: bạn sinh từng token một. Đưa `[t1, t2, t3]`, nhận `t4`. Đưa `[t1, t2, t3, t4]`, nhận `t5`. Đưa `[t1, t2, t3, t4, t5]`, nhận `t6`. KV cache (Bài 12) lưu trạng thái ẩn của `t1…tn` để bạn không phải tính lại chúng mỗi bước. Nhưng độ sâu tuần tự khi suy luận = độ dài đầu ra. Đó là thứ thuế autoregressive (tự hồi quy), và cũng là lý do giải mã là nút thắt độ trễ của mọi LLM.

### Mất mát — dịch một vị trí

Cho các token `[t1, t2, t3, t4]`:

- Đầu vào: `[t1, t2, t3]`
- Mục tiêu: `[t2, t3, t4]`

Với mọi vị trí `i`, tính `-log P(target_i | inputs[:i+1])`. Cộng lại. Đây là cross-entropy cho cả chuỗi.

Mọi transformer LM mà bạn từng nghe đều huấn luyện trên mất mát này. Pre-training, fine-tuning, SFT — cùng mất mát, khác dữ liệu.

### Các chiến lược giải mã

Sau khi huấn luyện, các lựa chọn lấy mẫu quan trọng hơn người ta tưởng.

| Phương pháp | Nó làm gì | Khi nào dùng |
|--------|--------------|-------------|
| Greedy | Argmax ở mỗi bước | Tác vụ tất định, hoàn thành code |
| Temperature | Chia logits cho T rồi lấy mẫu | Tác vụ sáng tạo, T cao hơn = đa dạng hơn |
| Top-k | Chỉ lấy mẫu từ k token đầu | Diệt các đuôi xác suất thấp |
| Top-p (nucleus) | Lấy mẫu từ tập nhỏ nhất có xác suất cộng dồn ≥ p | Mặc định từ 2020 trở đi; thích ứng theo hình dạng phân phối |
| Min-p | Giữ các token có `p > min_p * max_p` | Từ 2024 trở đi; loại đuôi dài tốt hơn top-p |
| Speculative decoding | Mô hình nháp đề xuất N token, mô hình lớn kiểm chứng | Giảm độ trễ 2–3× với chất lượng như cũ |

Năm 2026, min-p + temperature (nhiệt độ) 0.7 là mặc định hợp lý cho các mô hình open-weights. Speculative decoding là mức tối thiểu bắt buộc với mọi stack suy luận production.

### Điều gì làm "công thức GPT" thành công

1. **Decoder-only.** Không có chi phí phụ từ encoder. Mỗi lớp chỉ một lượt attention + FFN.
2. **Scaling.** 124M → 1.5B → 175B → hàng nghìn tỷ. Các định luật scaling của Chinchilla (Bài 13) cho bạn biết cách tiêu compute.
3. **In-context learning.** Xuất hiện vào khoảng 6B–13B. Mô hình có thể làm theo các ví dụ few-shot (học từ vài mẫu) mà không cần fine-tuning.
4. **RLHF.** Post-training trên sở thích của con người đã biến văn bản thô sau tiền huấn luyện thành trợ lý trò chuyện.
5. **Pre-norm + RoPE + SwiGLU.** Huấn luyện ổn định ở quy mô lớn.

Kiến trúc lõi không thay đổi nhiều kể từ GPT-2. Mọi thứ thú vị đều diễn ra ở dữ liệu, quy mô và post-training.

```figure
causal-mask
```

## Tự tay làm

### Bước 1: mặt nạ nhân quả

Xem `code/main.py`. Chỉ một dòng:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Cộng nó vào điểm attention trước softmax. Toàn bộ cơ chế chỉ có thế.

### Bước 2: một mô hình kiểu GPT 2 lớp

Xếp hai khối decoder (self-attention có mặt nạ + FFN, không có cross-attention). Thêm một token embedding, một positional encoding và một unembedding (buộc chung với ma trận token embedding — một thủ thuật chuẩn từ GPT-2).

### Bước 3: dự đoán token kế tiếp, end-to-end

Trên bộ từ vựng đồ chơi 20 token, sinh logits ở mọi vị trí. Tính mất mát cross-entropy so với mục tiêu dịch một vị trí. Không có gradient — đây là một kiểm tra nhanh cho lượt forward.

### Bước 4: lấy mẫu

Cài đặt greedy, temperature, top-k, top-p, min-p. Chạy từng cái trên một prompt cố định và so sánh đầu ra. Một hàm lấy mẫu chỉ tốn 10 dòng.

## Áp dụng

PyTorch, theo cách viết năm 2026:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Bên dưới, `generate()` chạy lượt forward, lấy logits ở vị trí cuối, lấy mẫu token kế tiếp, nối nó vào, rồi lặp lại. Mọi stack suy luận LLM production (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) đều cài đặt cùng vòng lặp đó với tối ưu nặng — prefill theo batch, continuous batching, phân trang KV cache, speculative decoding.

**GPT vs BERT, mỗi bên một dòng:** GPT dự đoán `P(x_t | x_{<t})`. BERT dự đoán `P(x_masked | x_unmasked)`. Mất mát quyết định mô hình có sinh được hay không.

## Đóng gói

Xem `outputs/skill-sampling-tuner.md`. Skill này chọn tham số lấy mẫu cho một tác vụ sinh mới và gắn cờ khi bắt buộc phải giải mã tất định.

## Bài tập

1. **Dễ.** Chạy `code/main.py` và kiểm chứng ma trận attention nhân quả là tam giác dưới sau softmax. Kiểm tra nhanh: hàng 3 chỉ nên có trọng số ở các cột 0–3.
2. **Trung bình.** Cài đặt beam search với độ rộng 4. So sánh perplexity giữa beam-4 và greedy trên 10 prompt ngắn. Beam có luôn thắng không? (Gợi ý: thường là thắng với dịch máy, không thắng với trò chuyện mở.)
3. **Khó.** Cài đặt speculative decoding: dùng một mô hình nhỏ 2 lớp làm mô hình nháp và một mô hình 6 lớp làm bộ kiểm chứng. Đo tốc độ theo đồng hồ thực trên 100 lượt hoàn thành dài 64. Xác nhận đầu ra khớp với greedy của bộ kiểm chứng.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Causal mask | "Cái tam giác" | Ma trận tam giác trên toàn `-inf` được cộng vào điểm attention để vị trí `i` chỉ thấy các vị trí `≤ i`. |
| Next-token prediction | "Cái mất mát" | Cross-entropy giữa phân phối của mô hình và token kế tiếp thật, ở mọi vị trí. |
| Autoregressive | "Sinh từng cái một" | Đưa đầu ra trở lại làm đầu vào; chỉ song song khi huấn luyện, không song song khi sinh. |
| Logits | "Điểm trước softmax" | Đầu ra thô của LM head trước softmax; việc lấy mẫu diễn ra trên các giá trị này. |
| Temperature | "Núm sáng tạo" | Chia logits cho T; T→0 = greedy, T→∞ = đều. |
| Top-p | "Lấy mẫu nucleus" | Cắt phân phối về tập nhỏ nhất có tổng ≥p; lấy mẫu từ phần còn lại. |
| Min-p | "Tốt hơn top-p" | Giữ các token có `p ≥ min_p × max_p`; ngưỡng cắt thích ứng theo độ nhọn của phân phối. |
| Speculative decoding | "Nháp + kiểm chứng" | Mô hình rẻ đề xuất N token; mô hình lớn kiểm chứng song song. |
| Teacher forcing | "Thủ thuật huấn luyện" | Khi huấn luyện, đưa vào token trước đó thật, không đưa dự đoán của mô hình. Chuẩn cho mọi seq2seq LM. |

## Đọc thêm

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 và in-context learning.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — bài báo về speculative decoding.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — mã tham chiếu chuẩn cho causal LM.

---

[← Bài trước: BERT](../../06-bert-masked-language-modeling/docs/vi.md) · [Bài sau: T5, BART →](../../08-t5-bart-encoder-decoder/docs/vi.md)
