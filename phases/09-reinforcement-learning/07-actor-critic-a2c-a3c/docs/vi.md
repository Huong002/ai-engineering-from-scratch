# Actor-Critic — A2C và A3C

> REINFORCE rất nhiễu. Thêm một critic (bộ phê bình) học `V̂(s)`, trừ nó khỏi return, và bạn có một lợi thế cùng kỳ vọng nhưng phương sai thấp hơn nhiều. Đó là actor-critic (tác tử–phê bình). A2C chạy nó đồng bộ; A3C chạy nó trên nhiều luồng. Cả hai là mô hình tư duy cho mọi phương pháp deep-RL hiện đại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 04 (Học TD), Giai đoạn 9, Bài 06 (REINFORCE)
**Time:** ~75 phút

## Vấn đề

REINFORCE nguyên bản vẫn chạy được, nhưng phương sai của nó rất tệ. Return Monte Carlo `G_t` có thể dao động tới 10 lần giữa các episode. Nhân độ nhiễu đó với `∇ log π` rồi lấy trung bình sẽ cho một bộ ước lượng gradient cần hàng nghìn episode để dịch policy đi được quãng đường mà vài trăm bước cập nhật DQN đã làm được.

Phương sai đến từ việc dùng return thô. Nếu bạn trừ đi một baseline `b(s_t)` — hàm bất kỳ của trạng thái, kể cả một value học được — kỳ vọng không đổi và phương sai giảm. Baseline tốt nhất mà còn tính được là `V̂(s_t)`. Giờ đại lượng nhân với `∇ log π` là *advantage* (lợi thế):

`A(s, a) = G - V̂(s)`

Một hành động là tốt nếu nó tạo ra return trên mức trung bình; xấu nếu dưới. REINFORCE với một critic học được chính là actor-critic. Critic trao cho actor một người thầy ít phương sai. Đây là mọi phương pháp deep-policy sau 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Khái niệm

![Actor-critic: mạng policy cộng mạng value, phần dư TD làm lợi thế](../assets/actor-critic.svg)

**Hai mạng, một loss chung:**

- **Actor** `π_θ(a | s)`: policy. Được lấy mẫu để hành động. Huấn luyện bằng policy gradient.
- **Critic** `V_φ(s)`: ước lượng return kỳ vọng từ trạng thái. Huấn luyện để cực tiểu `(V_φ(s) - target)²`.

**Hàm lợi thế (advantage function).** Hai dạng chuẩn:

- *Lợi thế MC:* `A_t = G_t - V_φ(s_t)`. Không bias, phương sai cao hơn.
- *Lợi thế TD:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Có bias (dùng `V_φ`), phương sai thấp hơn nhiều. Còn gọi là *phần dư TD* `δ_t`.

**Lợi thế n bước.** Nội suy giữa hai dạng trên:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` là TD thuần. `n = ∞` là MC. Hầu hết implementation dùng `n = 5` cho Atari, `n = 2048` cho PPO trên MuJoCo.

**Generalized Advantage Estimation (GAE).** Schulman et al. (2016) đề xuất trung bình có trọng số mũ trên toàn bộ các lợi thế n bước:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

với `λ ∈ [0, 1]`. `λ = 0` là TD (phương sai thấp, bias cao). `λ = 1` là MC (phương sai cao, không bias). `λ = 0.95` là mặc định năm 2026 — chỉnh tới khi núm bias/phương sai nằm đúng chỗ bạn muốn.

**A2C: advantage actor-critic đồng bộ.** Thu `T` bước trên `N` môi trường song song. Tính lợi thế cho từng bước. Cập nhật actor và critic trên batch gộp lại. Lặp lại. Đây là người anh em đơn giản hơn, dễ mở rộng hơn của A3C.

**A3C: advantage actor-critic bất đồng bộ.** Mnih et al. (2016). Sinh ra `N` worker thread (tiến trình song song), mỗi cái chạy một môi trường. Mỗi worker tính gradient cục bộ trên rollout (lượt chạy thu dữ liệu) của riêng nó, rồi bất đồng bộ áp chúng lên một parameter server dùng chung. Không cần replay buffer — các worker tự khử tương quan bằng cách chạy những quỹ đạo khác nhau. A3C đã chứng minh bạn có thể huấn luyện trên CPU ở quy mô lớn. Năm 2026, A2C trên GPU (môi trường song song dạng batch) chiếm ưu thế vì GPU cần batch lớn.

**Loss gộp.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Ba hạng tử: loss policy gradient, hồi quy value, thưởng entropy (entropy). `c_v ~ 0.5`, `c_e ~ 0.01` là những điểm khởi đầu chuẩn.

```figure
actor-critic
```

## Tự tay làm

### Bước 1: một critic

Critic tuyến tính `V_φ(s) = w · features(s)` cập nhật bằng MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Trên môi trường dạng bảng, critic hội tụ sau vài trăm episode. Trên Atari, thay critic tuyến tính bằng thân CNN dùng chung + đầu value.

### Bước 2: lợi thế n bước

Cho một rollout độ dài `T` và `V(s_T)` cuối đã bootstrapping (lấy ước lượng để khép tổng):

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` là target cho critic. `advantages` là cái nhân với `∇ log π`.

### Bước 3: cập nhật gộp

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy, một rollout cho mỗi lần cập nhật, learning rate riêng cho actor và critic.

### Bước 4: song song hoá (A3C vs A2C)

- **A3C:** dựng `N` thread. Mỗi thread chạy môi trường riêng và lượt forward riêng. Định kỳ đẩy cập nhật gradient lên master dùng chung. Không khoá master — race là chấp nhận được, chúng chỉ thêm nhiễu.
- **A2C:** chạy `N` instance môi trường trong một tiến trình, xếp observation thành batch `[N, obs_dim]`, forward theo batch, backward theo batch. Tận dụng GPU tốt hơn, tất định, dễ suy luận hơn. Lựa chọn mặc định năm 2026.

Code đồ chơi của chúng ta chạy đơn luồng cho rõ ràng; viết lại thành A2C dạng batch chỉ mất ba dòng numpy.

## Cạm bẫy

- **Critic còn bias trước khi có gradient của actor.** Nếu critic còn ngẫu nhiên, baseline của nó vô nghĩa và bạn đang huấn luyện trên nhiễu thuần. Làm nóng critic vài trăm bước trước khi bật policy gradient, hoặc dùng learning rate chậm cho actor.
- **Chuẩn hoá lợi thế.** Chuẩn hoá lợi thế về zero-mean/unit-std theo từng batch. Ổn định huấn luyện rất nhiều với chi phí gần bằng không.
- **Thân dùng chung.** Với đầu vào ảnh, dùng một bộ trích đặc trưng dùng chung cho actor và critic. Hai đầu riêng biệt. Đặc trưng dùng chung được hưởng lợi từ cả hai loss.
- **Hợp đồng on-policy.** A2C dùng lại dữ liệu đúng một lần cập nhật. Nhiều hơn là gradient bị bias (hiệu chỉnh importance-sampling chính là thứ PPO thêm vào).
- **Sụp đổ entropy.** Không có `c_e > 0`, policy gần như tất định sau vài trăm lần cập nhật và ngừng khám phá.
- **Thang reward.** Độ lớn của lợi thế phụ thuộc thang reward. Chuẩn hoá reward (ví dụ chia theo std chạy) để độ lớn gradient nhất quán giữa các task.

## Áp dụng

Năm 2026 A2C/A3C hiếm khi là lựa chọn cuối cùng, nhưng chúng là kiến trúc mà mọi thứ về sau tinh chỉnh:

| Phương pháp | Quan hệ với A2C |
|--------|----------------|
| PPO | A2C + tỉ lệ importance bị cắt cho cập nhật nhiều epoch |
| IMPALA | A3C + hiệu chỉnh off-policy V-trace |
| SAC (Giai đoạn 9 · 07) | A2C off-policy với critic soft-value (bài sau) |
| GRPO (Giai đoạn 9 · 12) | A2C không có critic — lợi thế tương đối theo nhóm |
| DPO | A2C thu gọn thành loss xếp hạng ưu tiên, không lấy mẫu |
| AlphaStar / OpenAI Five | A2C với huấn luyện theo liên minh + tiền huấn luyện bắt chước |

Nếu bạn thấy "advantage" trong một paper năm 2026, hãy nghĩ tới actor-critic.

## Đóng gói

Lưu thành `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Bài tập

1. **Dễ.** Huấn luyện actor-critic với lợi thế MC (`G_t - V(s_t)`) trên GridWorld 4×4. So sánh hiệu quả mẫu với REINFORCE-dùng-baseline-trung-bình-chạy ở Bài 06.
2. **Trung bình.** Chuyển sang lợi thế phần dư TD (`r + γ V(s') - V(s)`). Đo phương sai của các batch lợi thế. Nó giảm bao nhiêu?
3. **Khó.** Cài GAE(λ). Quét `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Vẽ return cuối theo hiệu quả mẫu. Điểm ngọt bias/phương sai cho task này nằm ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Actor | "Mạng policy" | `π_θ(a\|s)`, cập nhật bằng policy gradient. |
| Critic | "Mạng value" | `V_φ(s)`, cập nhật bằng hồi quy MSE tới return / target TD. |
| Advantage | "Tốt hơn mức trung bình bao nhiêu" | `A(s, a) = Q(s, a) - V(s)` hoặc các bộ ước lượng của nó. Hệ số nhân cho `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; ước lượng lợi thế một bước. |
| GAE | "Núm chỉnh nội suy" | Tổng có trọng số mũ của các lợi thế n bước, tham số hoá bởi `λ`. |
| A2C | "Actor-critic đồng bộ" | Gộp batch qua các môi trường; một bước gradient mỗi rollout. |
| A3C | "Actor-critic bất đồng bộ" | Các worker thread đẩy gradient lên param server dùng chung. Paper gốc; ít phổ biến hơn năm 2026. |
| Bootstrap | "Dùng V ở chân trời" | Cắt rollout, cộng `γ^n V(s_{t+n})` để khép tổng. |

## Đọc thêm

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, paper actor-critic bất đồng bộ gốc.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — nền tảng; ghép với Ch. 9 về xấp xỉ hàm khi critic là mạng nơ-ron.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — actor-critic phân tán dễ mở rộng với hiệu chỉnh off-policy V-trace.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — các implementation A2C/PPO chạy production, đáng đọc.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — kết quả hội tụ nền tảng cho cách tách actor-critic hai thang thời gian.

---

[← Bài trước: Policy Gradient](../../06-policy-gradients-reinforce/docs/vi.md) · [Bài sau: Tối ưu hoá chính sách lân cận (Proximal Policy Optimization, PPO) →](../../08-ppo/docs/vi.md)
