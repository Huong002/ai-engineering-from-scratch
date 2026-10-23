# Giải mã suy đoán và EAGLE

> Một LLM tiên tiến sinh ra một token cần một lượt forward pass đầy đủ qua hàng tỉ tham số. Lượt forward pass đó được cấp năng lực dư thừa rất nhiều: phần lớn thời gian, một mô hình nhỏ hơn nhiều có thể đoán đúng 3-5 token tiếp theo, và mô hình lớn chỉ cần *kiểm chứng* phỏng đoán đó. Khi phỏng đoán đúng, bạn có 5 token với giá của một token. Speculative decoding (giải mã suy đoán), công trình của Leviathan et al. 2023, đã làm điều này thành chính xác, và EAGLE-3 (2025) đẩy acceptance rate (tỉ lệ chấp nhận) lên ~4.5 token mỗi lần kiểm chứng — speedup (hệ số tăng tốc) 4-5 lần với phân phối đầu ra khớp.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Giai đoạn 10, Bài 12 (tối ưu suy luận), Giai đoạn 10, Bài 04 (huấn luyện trước mini-GPT)
**Time:** ~75 phút

## Vấn đề

Thông lượng giải mã của một mô hình cỡ 70B trên H100 thường là 40-80 token/giây. Mỗi token cần một lượt forward pass đầy đủ đọc toàn bộ trọng số mô hình từ HBM. Bạn không thể làm mô hình nhỏ đi mà không thay đổi đầu ra của nó. Bạn không thể tăng batch size vượt quá giới hạn bộ nhớ. Bạn bị kẹt — trừ khi có thể để mô hình xuất ra nhiều hơn một token mỗi lượt forward pass.

Sinh tự hồi quy (autoregressive) trông có vẻ tuần tự sẵn: `x_{t+1} = sample(p(· | x_{1:t}))`. Nhưng có một cơ hội song song hoá. Nếu bạn có một bộ dự đoán rẻ nói rằng "4 token tiếp theo có lẽ là [a, b, c, d]" thì bạn có thể kiểm chứng cả 5 vị trí trong **một lượt forward pass duy nhất của mô hình lớn** và chấp nhận tiền tố khớp dài nhất.

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding") đã làm điều này thành chính xác nhờ một quy tắc chấp nhận/từ chối khéo léo giữ nguyên phân phối lấy mẫu của mô hình đích. Cùng một phân phối đầu ra, nhanh hơn 2-4×.

## Khái niệm

### Thiết lập hai mô hình

- **Target model** (mô hình đích) `M_p`: mô hình lớn, chậm, chất lượng cao mà bạn thực sự muốn lấy mẫu từ đó. Phân phối: `p(x)`.
- **Draft model** (mô hình nháp) `M_q`: một mô hình nhỏ, nhanh, chất lượng thấp hơn. Phân phối: `q(x)`. Nhỏ hơn 5-30 lần.

Mỗi bước:

1. Mô hình nháp đề xuất `K` token một cách tự hồi quy: `x_1, x_2, ..., x_K ~ q`.
2. Mô hình đích chạy MỘT lượt forward pass trên cả `K+1` vị trí song song, sinh ra `p(x_k)` cho từng token được đề xuất.
3. Chấp nhận/từ chối từng token từ trái sang phải theo quy tắc rejection sampling (lấy mẫu loại bỏ) đã sửa ở dưới. Chấp nhận tiền tố khớp dài nhất.
4. Nếu có token bị từ chối, lấy mẫu token thay thế từ phân phối đã hiệu chỉnh rồi dừng. Ngược lại, lấy mẫu thêm một token thưởng từ `p(· | x_1...x_K)`.

Nếu mô hình nháp khớp hoàn hảo với mô hình đích, bạn được K+1 token mỗi lượt forward của mô hình đích. Nếu mô hình nháp sai ngay ở vị trí 1, bạn chỉ được 1 token.

### Quy tắc chính xác

Giải mã suy đoán **được chứng minh là tương đương về phân phối với việc lấy mẫu từ p** — tức không mất mát (lossless). Quy tắc từ chối:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

trong đó `(p - q)+` là phần dương của hiệu từng điểm. Khi mô hình nháp và mô hình đích khớp nhau (`p ≈ q`) thì xác suất chấp nhận gần bằng 1. Khi chúng khác nhau, phân phối phần dư được dựng sao cho mẫu tổng thể vẫn chính xác là `p`.

**Trường hợp tham lam (greedy).** Với lấy mẫu temperature=0, chỉ cần kiểm tra `argmax(p) == x_t`. Nếu đúng thì chấp nhận; nếu không thì xuất `argmax(p)` và dừng.

### Hệ số tăng tốc kỳ vọng

Nếu acceptance rate ở cấp token của mô hình nháp là `α`, số token kỳ vọng sinh ra mỗi lượt forward pass của mô hình đích là:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

Với `α = 0.8, K = 4`: `(1 - 0.8^5)/(1 - 0.8) = 3.36` token mỗi lượt forward. Một lượt forward của mô hình đích tốn khoảng `cost_q * K + cost_p` (K bước nháp cộng một lần kiểm chứng của mô hình đích). Nếu `cost_p >> cost_q * K` thì tỉ lệ speedup là `3.36× / 1 = 3.36×` về thông lượng.

Tham số thực sự duy nhất là `α`, và nó phụ thuộc hoàn toàn vào mức khớp giữa nháp và đích. Một mô hình nháp tốt là tất cả.

### Huấn luyện mô hình nháp: chưng cất

Một mô hình nhỏ ngẫu nhiên làm nháp rất kém. Công thức chuẩn là chưng cất (distill) từ mô hình đích:

1. Chọn một kiến trúc nhỏ (~1B cho đích 70B, ~500M cho đích 7B).
2. Chạy mô hình đích trên một kho văn bản lớn; lưu lại phân phối token tiếp theo của nó.
3. Huấn luyện mô hình nháp bằng KL divergence so với phân phối của mô hình đích (không phải so với token đúng thực tế).

Kết quả: `α` thường 0.6-0.8 với lập trình, 0.7-0.85 với chat ngôn ngữ tự nhiên. Tăng tốc 2-3× trong môi trường production.

### EAGLE: soạn cây + tái dùng đặc trưng

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") chỉ ra hai điểm kém hiệu quả trong giải mã suy đoán tiêu chuẩn:

1. Mô hình nháp chạy K bước tuần tự, bước nào cũng qua toàn bộ chồng lớp. Nhưng mô hình nháp có thể tái dùng đặc trưng (hidden states) của mô hình đích từ lần kiểm chứng gần nhất — mô hình đích đã tính ra những biểu diễn phong phú mà mô hình nháp đang tính lại từ đầu.
2. Mô hình nháp xuất ra một chuỗi tuyến tính. Nếu mô hình nháp có thể xuất ra một *cây* ứng viên (mỗi nút nhiều phỏng đoán), thì lượt forward pass duy nhất của mô hình đích có thể kiểm chứng nhiều đường ứng viên song song nhờ mặt nạ attention dạng cây, rồi chọn nhánh được chấp nhận dài nhất.

Những thay đổi của EAGLE-1:
- Đầu vào của mô hình nháp = hidden state cuối cùng của mô hình đích tại vị trí t, không phải token thô.
- Kiến trúc nháp = 1 lớp transformer decoder (không phải một mô hình nhỏ riêng).
- Đầu ra = cây gồm K = 4-8 ứng viên mỗi độ sâu, độ sâu 4-6.

EAGLE-2 (2024) thêm cấu trúc cây động: cây mở rộng hơn ở chỗ mô hình nháp không chắc chắn và giữ hẹp ở chỗ nó tự tin. Cách này nâng `α_effective` mà không tăng chi phí kiểm chứng.

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") bỏ phụ thuộc cố định vào đặc trưng lớp trên cùng và huấn luyện mô hình nháp bằng một loss mới "test-time simulation" — mô hình nháp được huấn luyện trên các đầu ra khớp phân phối lúc suy luận của mô hình đích thay vì phân phối lúc huấn luyện kiểu teacher forcing. Tỉ lệ chấp nhận tăng từ 0.75 (EAGLE-2) lên 0.82 (EAGLE-3), và số token trung bình mỗi lần kiểm chứng từ 3.0 lên 4.5.

### Kiểm chứng bằng tree attention

Khi mô hình nháp xuất ra một cây, mô hình đích kiểm chứng nó trong một lượt forward pass bằng **mặt nạ attention dạng cây (tree attention mask)** — một mask nhân quả mã hoá cấu trúc cây thay vì một đường thẳng. Mỗi token chỉ attend tới các tổ tiên của nó trong cây. Lượt kiểm chứng vẫn là một forward, một matmul; mask cấu trúc chỉ tốn thêm vài mục KV.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Nếu `a, b` là các ứng viên token đầu cạnh tranh nhau và `c, d, e, f` là các ứng viên token thứ hai, cả sáu vị trí được kiểm chứng trong một lượt forward pass. Đầu ra là tiền tố dài nhất dọc theo bất kỳ đường nào được chấp nhận.

### Khi nào thắng, khi nào không

**Thắng:**
- Chat / hoàn thành văn bản có tính dự đoán được (code, tiếng Anh thông dụng, đầu ra có cấu trúc). `α` cao.
- Những bối cảnh còn compute GPU nhàn rỗi trong lúc giải mã (pha bị giới hạn bởi bộ nhớ). Soạn cây dùng phần FLOPs còn trống.

**Thua / không lợi:**
- Đầu ra rất ngẫu nhiên (viết sáng tạo ở temperature cao). `α` tụt về `1/|vocab|`.
- Phục vụ theo batch với độ đồng thời rất cao — batching đã lấp đầy FLOPs, còn ít chỗ cho kiểm chứng cây.
- Mô hình đích rất nhỏ, khi mô hình nháp không nhỏ hơn bao nhiêu.

Các xưởng production thường báo tăng tốc 2-3× thời gian thực trên chat, 3-5× trên sinh code, và gần như bằng không với viết sáng tạo.

```figure
speculative-decoding
```

## Tự tay làm

`code/main.py`:

- Một `speculative_decode(target, draft, prompt, K, temperature)` tham chiếu, cài đúng quy tắc từ chối chính xác và kiểm chứng rằng nó giữ nguyên phân phối của mô hình đích (KL thực nghiệm < 0.01 so với lấy mẫu thuần từ mô hình đích).
- Một bộ soạn cây kiểu EAGLE dựng cây độ sâu K với phân nhánh top-p.
- Một bộ dựng mặt nạ attention dạng cây tạo đúng mẫu nhân quả cho bộ kiểm chứng.
- Một harness đo acceptance rate chạy cả hai trên một LM tí hon (chưng cất một GPT-2-small từ đích GPT-2-medium).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Áp dụng

- **vLLM** và **SGLang** có sẵn giải mã suy đoán như một tính năng hạng nhất. Trong vLLM, truyền cho `--speculative-config` một đối tượng JSON với `method`, `model`, và `num_speculative_tokens`; EAGLE-3 là `"method": "eagle3"`.
- **NVIDIA TensorRT-LLM** hỗ trợ cây Medusa và EAGLE một cách gốc.
- **Mô hình nháp tham chiếu**: `Qwen/Qwen3-0.6B` (nháp cho Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct` (nháp cho Llama 3.x 70B).
- **Medusa heads** (Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): thay vì một mô hình nháp, thêm K đầu dự đoán song song vào chính mô hình đích. Triển khai đơn giản hơn, acceptance rate thấp hơn EAGLE một chút.

## Đóng gói

Bài học này tạo ra `outputs/skill-speculative-tuning.md` — một skill khảo sát khối lượng công việc của mô hình đích và chọn: mô hình nháp, K (độ dài nháp), độ rộng cây, temperature, và khi nào quay lại giải mã thường.

## Bài tập

1. Cài đặt quy tắc từ chối chính xác và kiểm chứng nó bằng thực nghiệm. Chạy 10K mẫu qua `speculative_decode` và qua lấy mẫu thuần từ mô hình đích; tính khoảng cách TV giữa hai phân phối đầu ra. Phải < 0.01.

2. Tính công thức speedup. Với `α` và `K` cố định, vẽ số token kỳ vọng mỗi lượt forward của mô hình đích. Tìm K tối ưu cho α ∈ {0.5, 0.7, 0.9}.

3. Huấn luyện một mô hình nháp tí hon. Lấy đích GPT-2 124M và chưng cất một nháp GPT-2 30M trên 100M token với loss KL. Đo `α` trên văn bản held-out. Kỳ vọng: 0.6-0.7.

4. Cài đặt soạn cây kiểu EAGLE. Thay vì một chuỗi, cho mô hình nháp xuất top-3 nhánh ở mỗi độ sâu. Dựng mặt nạ attention dạng cây. Kiểm chứng rằng mô hình đích chấp nhận nhánh đúng dài nhất.

5. Đo các chế độ thất bại. Chạy giải mã suy đoán ở temperature=1.5 (rất ngẫu nhiên). Cho thấy α sụp và thuật toán chậm hơn giải mã thường vì chi phí phụ của mô hình nháp.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Target model | "Mô hình lớn" | Mô hình chậm, chất lượng cao mà bạn muốn lấy mẫu từ đó (phân phối p) |
| Draft model | "Bộ suy đoán" | Bộ dự đoán nhỏ, nhanh (phân phối q); nhỏ hơn 5-30 lần |
| K / draft length | "Nhìn trước" | Số token được suy đoán mỗi lượt kiểm chứng |
| α / acceptance rate | "Tỉ lệ trúng" | Xác suất, ở mức mỗi token, rằng đề xuất của mô hình nháp được chấp nhận |
| Exact rejection rule | "Phép thử chấp nhận" | So sánh r < p/q giữ nguyên phân phối của mô hình đích |
| Residual distribution | "p-q đã hiệu chỉnh" | (p - q)+ / ||(p - q)+||_1, phân phối để lấy mẫu khi từ chối |
| Tree drafting | "Suy đoán phân nhánh" | Mô hình nháp xuất một cây ứng viên, kiểm chứng trong một lượt bằng mặt nạ attention cấu trúc cây |
| Tree attention mask | "Mặt nạ cấu trúc" | Mask nhân quả mã hoá cấu trúc cây để mỗi nút chỉ attend tới các tổ tiên của nó |
| Medusa heads | "Đầu song song" | K đầu dự đoán thêm trên chính mô hình đích; không cần mô hình nháp riêng |
| EAGLE feature reuse | "Nháp từ hidden state" | Đầu vào của nháp là hidden state cuối của mô hình đích, không phải token thô, nhờ đó nháp nhỏ hơn |
| Test-time simulation loss | "Huấn luyện EAGLE-3" | Huấn luyện nháp trên đầu ra khớp phân phối lúc suy luận của đích, không dùng teacher forcing |

## Đọc thêm

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — quy tắc từ chối chính xác và phân tích tăng tốc lý thuyết
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — bài báo speculative sampling song song tại DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — phương án đầu song song thay cho mô hình nháp
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — tái dùng đặc trưng và soạn cây
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — cấu trúc cây động
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — khớp phân phối lúc huấn luyện với lúc suy luận
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — giải mã Jacobi/lookahead, một lựa chọn không cần bộ suy đoán

---

[← Bài trước: Suy luận bất đồng bộ và Hogwild!](../../22-async-hogwild-inference/docs/vi.md) · [Bài sau: Checkpoint gradient và tính lại hoạt hoá →](../../34-gradient-checkpointing/docs/vi.md)
