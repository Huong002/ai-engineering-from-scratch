# Tối ưu hoá chính sách lân cận (Proximal Policy Optimization, PPO)

> A2C vứt bỏ mỗi cuộn dữ liệu sau một lần cập nhật. PPO bọc policy gradient trong một importance sampling ratio (tỉ lệ lấy mẫu quan trọng) đã bị clipping (cắt biên) để bạn có thể chạy 10+ epoch trên cùng dữ liệu mà chính sách không nổ tung. Schulman et al. (2017). Đến 2026 vẫn là thuật toán policy-gradient mặc định.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 06 (REINFORCE), Giai đoạn 9, Bài 07 (Actor-Critic)
**Time:** ~75 phút

## Vấn đề

A2C (Bài 07) là on-policy: gradient `E_{π_θ}[A · ∇ log π_θ]` đòi hỏi dữ liệu lấy mẫu từ `π_θ` *hiện tại*. Cập nhật một lần là `π_θ` đổi; dữ liệu bạn vừa dùng giờ thành off-policy. Dùng lại thì gradient của bạn bị chệch.

Rollout (cuộn dữ liệu) rất đắt. Trên Atari, một cuộn qua 8 env × 128 bước = 1024 transition và cả chục giây thời gian môi trường. Vứt bỏ nó sau một bước gradient là lãng phí.

Trust Region Policy Optimization (vùng tin cậy, TRPO, Schulman 2015) là cách sửa đầu tiên: ràng buộc mỗi cập nhật sao cho KL divergence giữa chính sách cũ và mới nằm dưới `δ`. Sạch về lý thuyết, nhưng cần giải conjugate-gradient mỗi lần cập nhật. Năm 2026 không ai chạy TRPO nữa.

PPO (Schulman et al. 2017) thay ràng buộc vùng tin cậy cứng bằng một mục tiêu bị clipping đơn giản. Thêm đúng một dòng code. Mười epoch mỗi cuộn. Không conjugate gradient. Bảo đảm lý thuyết đủ tốt. Chín năm sau, nó vẫn là thuật toán policy-gradient mặc định cho mọi thứ từ MuJoCo đến RLHF.

## Khái niệm

![Mục tiêu thay thế bị clipping của PPO: cắt tỉ lệ tại 1 ± ε](../assets/ppo.svg)

**Tỉ lệ lấy mẫu quan trọng.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Đây là tỉ lệ hợp lý của chính sách mới so với chính sách đã thu dữ liệu. `r_t = 1` nghĩa là không đổi. `r_t = 2` nghĩa là chính sách mới có xác suất chọn `a_t` gấp đôi chính sách cũ.

**Surrogate objective (mục tiêu thay thế) bị clipping.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Hai số hạng:

- Nếu advantage (lợi thế) `A_t > 0` và tỉ lệ muốn vượt quá `1 + ε`, phép clip làm phẳng gradient — đừng đẩy một hành động tốt xa hơn `+ε` so với xác suất cũ.
- Nếu advantage `A_t < 0` và tỉ lệ muốn vượt quá `1 - ε` (nghĩa là ta sẽ khiến một hành động xấu có khả năng xảy ra cao hơn so với mức giảm đã bị clip của nó), phép clip chặn gradient — đừng đẩy một hành động xấu xuống dưới `-ε`.

`min` xử lý hướng còn lại: nếu tỉ lệ đã di chuyển theo hướng *có lợi*, bạn vẫn nhận gradient (không clip ở phía sẽ làm hại bạn).

`ε = 0.2` là giá trị điển hình. Vẽ mục tiêu như hàm của `r_t`: một hàm tuyến tính từng khúc với mái bằng ở "phía tốt" và sàn bằng ở "phía xấu".

**Toàn bộ loss của PPO.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Cùng cấu trúc actor-critic như A2C. Ba hệ số, thường là `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Vòng lặp huấn luyện.**

1. Thu `N × T` transition qua `N` env song song, mỗi env `T` bước.
2. Tính advantage (GAE), đóng băng chúng thành hằng số.
3. Đóng băng `π_{θ_old}` như một ảnh chụp của `π_θ` hiện tại.
4. Với `K` epoch, với mỗi minibatch của `(s, a, A, V_target, log π_old(a|s))`:
   - Tính `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Áp dụng `L^{CLIP}` + value loss + entropy.
   - Bước gradient.
5. Vứt bỏ cuộn dữ liệu. Quay lại bước 1.

`K = 10` và minibatch 64 là một bộ siêu tham số chuẩn. PPO rất vững: các con số chính xác hiếm khi quan trọng trong khoảng ±50%.

**Biến thể phạt KL (KL penalty).** Bài báo gốc đề xuất một cách khác dùng phạt KL thích ứng: `L = L^{PG} - β · KL(π_θ || π_old)` với `β` được điều chỉnh dựa trên KL quan sát được. Phiên bản clipping trở nên thống trị; biến thể KL còn sống trong RLHF (nơi KL so với chính sách tham chiếu vốn là một ràng buộc riêng mà bạn luôn muốn có).

```figure
ppo-clip
```

## Tự tay làm

### Bước 1: chụp lại `log π_old(a | s)` tại lúc rollout

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Ảnh chụp được lấy một lần, ngay tại thời điểm rollout. Nó không đổi trong các epoch cập nhật.

### Bước 2: tính advantage GAE (Bài 07)

Giống A2C. Chuẩn hoá trên toàn batch.

### Bước 3: cập nhật mục tiêu thay thế bị clipping

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Mẫu "bị clip → gradient bằng 0" là trái tim của PPO. Nếu chính sách mới đã trôi quá xa theo hướng có lợi, cập nhật dừng lại.

### Bước 4: value và entropy

Thêm MSE chuẩn vào mục tiêu của critic và một phần thưởng entropy cho actor, giống A2C.

### Bước 5: chẩn đoán

Ba thứ cần theo dõi ở mỗi lần cập nhật:

- **KL trung bình** `E[log π_old - log π_θ]`. Nên nằm trong `[0, 0.02]`. Nếu vượt quá `0.1`, giảm `K_EPOCHS` hoặc `LR`.
- **Clip fraction** — phần mẫu có tỉ lệ nằm ngoài `[1-ε, 1+ε]`. Nên là `~0.1-0.3`. Nếu `~0`, clip không bao giờ kích hoạt → tăng `LR` hoặc `K_EPOCHS`. Nếu `~0.5+`, bạn đang overfit cuộn dữ liệu → giảm chúng.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Thước đo chất lượng critic. Nên tăng dần về 1 khi critic học.

## Cạm bẫy

- **Hệ số clip chỉnh sai.** `ε = 0.2` là chuẩn thực tế. Xuống `0.1` khiến cập nhật quá rụt rè; `0.3+` mời gọi bất ổn.
- **Quá nhiều epoch.** `K > 20` thường làm mất ổn định vì chính sách trôi xa `π_old`. Giới hạn epoch, đặc biệt với mạng lớn.
- **Không chuẩn hoá reward.** Thang reward lớn ăn vào khoảng clip. Chuẩn hoá reward (std chạy) trước khi tính advantage.
- **Quên chuẩn hoá advantage.** Chuẩn hoá zero-mean/unit-std theo batch là chuẩn mực. Bỏ qua sẽ phá PPO trên hầu hết benchmark.
- **Learning rate không giảm dần.** PPO hưởng lợi từ LR decay tuyến tính về 0. LR hằng thường tệ hơn.
- **Sai toán tỉ lệ.** Luôn dùng `exp(log_new - log_old)` để ổn định số học, không dùng `new / old`.
- **Sai dấu gradient.** Cực đại hoá mục tiêu thay thế = *cực tiểu hoá* `-L^{CLIP}`. Đảo dấu là lỗi PPO phổ biến nhất.

## Áp dụng

PPO là thuật toán RL mặc định của năm 2026 ở một số lượng đáng ngạc nhiên các lĩnh vực:

| Trường hợp dùng | Biến thể PPO |
|----------|-------------|
| Điều khiển robot / MuJoCo | PPO với chính sách Gaussian, GAE(0.95) |
| Atari / game rời rạc | PPO với chính sách categorical, cuộn 128 bước trượt dần |
| RLHF cho LLM | PPO với phạt KL so với mô hình tham chiếu, reward từ RM ở cuối câu trả lời |
| Agent game quy mô lớn | IMPALA + PPO (AlphaStar, OpenAI Five) |
| LLM suy luận | GRPO (Bài 12) — biến thể PPO không có critic |
| Dữ liệu chỉ có sở thích | DPO — dạng đóng thu gọn của PPO+KL, không lấy mẫu online |

*Hình dạng loss* của PPO — mục tiêu thay thế bị clipping + value + entropy — là giàn giáo cho DPO, GRPO và gần như mọi pipeline RLHF.

## Đóng gói

Lưu thành `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Bài tập

1. **Dễ.** Chạy PPO trên GridWorld 4×4 với `ε=0.2, K=4`. So sánh hiệu quả mẫu với A2C (một epoch mỗi cuộn) ở cùng số env step.
2. **Trung bình.** Quét `K ∈ {1, 4, 10, 30}`. Vẽ return theo env step và theo dõi KL trung bình mỗi lần cập nhật. Với `K` nào thì KL nổ trên bài toán này?
3. **Khó.** Thay mục tiêu thay thế bị clipping bằng phạt KL thích ứng (`β` gấp đôi nếu `KL > 2·target`, giảm nửa nếu `KL < target/2`). So sánh return cuối, độ ổn định và việc không cần clip.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; độ lệch so với chính sách đã thu dữ liệu. |
| Clipped surrogate | "chiêu chính của PPO" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; gradient phẳng khi vượt mức clip ở phía có lợi. |
| Trust region | "ý đồ của TRPO / PPO" | Giới hạn KL của mỗi cập nhật để bảo đảm cải thiện đơn điệu. |
| KL penalty | "vùng tin cậy mềm" | PPO bản khác: `L - β · KL(π_θ \|\| π_old)`. `β` thích ứng. |
| Clip fraction | "tần suất clip kích hoạt" | Chẩn đoán — nên là 0.1-0.3; ngoài khoảng đó là chỉnh sai. |
| Multi-epoch training | "dùng lại dữ liệu" | K epoch trên mỗi cuộn; đánh đổi phương sai lấy hiệu quả mẫu. |
| On-policy-ish | "gần như on-policy" | PPO danh nghĩa là on-policy nhưng K>1 epoch dùng dữ liệu hơi lệch chính sách một cách an toàn. |
| PPO-KL | "PPO còn lại" | Biến thể phạt KL; dùng trong RLHF nơi KL so với tham chiếu vốn đã là ràng buộc. |

## Đọc thêm

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — bài báo gốc.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, tiền thân của PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — từng siêu tham số của PPO được khảo sát riêng.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; công thức PPO trong RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — trình bày hiện đại, rõ ràng kèm PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — PPO một file để tham chiếu, được nhiều bài báo dùng.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — công thức production cho PPO trên mô hình ngôn ngữ; đọc cùng Bài 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — bài "37 tối ưu cấp code"; mẹo PPO nào là chịu lực và mẹo nào là truyền miệng.

---

[← Bài trước: Actor-Critic](../../07-actor-critic-a2c-a3c/docs/vi.md) · [Bài sau: Mô hình hóa phần thưởng & RLHF →](../../09-reward-modeling-rlhf/docs/vi.md)
