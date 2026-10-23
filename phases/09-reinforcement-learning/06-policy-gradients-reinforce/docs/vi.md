# Policy Gradient — REINFORCE từ đầu

> Đừng ước lượng giá trị nữa. Hãy tham số hóa chính sách một cách trực tiếp, tính gradient của lợi tức (return — tổng phần thưởng tích lũy) kỳ vọng, rồi bước ngược dốc. Williams (1992) đã viết điều này gọn trong một định lý. Đó là lý do PPO, GRPO và mọi vòng RL của LLM tồn tại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 3, Bài 03 (Lan truyền ngược), Giai đoạn 9, Bài 03 (Monte Carlo), Giai đoạn 9, Bài 04 (Học TD)
**Time:** ~75 phút

## Vấn đề

Q-learning và DQN tham số hóa hàm *giá trị*. Bạn chọn hành động bằng `argmax Q`. Cách đó ổn với hành động rời rạc và trạng thái rời rạc. Nó vỡ khi hành động liên tục (lấy `argmax` trên mô-men 10 chiều kiểu gì?) hoặc khi bạn muốn một chính sách ngẫu nhiên (`argmax` vốn dĩ tất định).

Gradient chính sách (policy gradient) thay vào đó tham số hóa *chính sách*. `π_θ(a | s)` là một mạng nơ-ron xuất ra phân phối trên các hành động. Lấy mẫu từ đó để hành động. Tính gradient của lợi tức kỳ vọng theo `θ`. Bước ngược dốc. Không `argmax`. Không đệ quy Bellman. Chỉ là leo gradient trên `J(θ) = E_{π_θ}[G]`.

Định lý REINFORCE (Williams 1992) nói rằng gradient này tính được: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Chạy một episode. Tính lợi tức. Nhân với `∇ log π_θ(a | s)` ở mỗi bước. Lấy trung bình. Leo gradient. Xong.

Mọi thuật toán RL-LLM năm 2026 — PPO, DPO, GRPO — đều là bản tinh chỉnh của REINFORCE. Hiểu nó bằng tay là điều kiện tiên quyết cho phần còn lại của giai đoạn này, cũng như cho Giai đoạn 10 · 07 (triển khai RLHF) và Giai đoạn 10 · 08 (DPO).

## Khái niệm

![Gradient chính sách: chính sách softmax, gradient log-π, cập nhật có trọng số theo lợi tức](../assets/policy-gradient.svg)

**Định lý gradient chính sách.** Với mọi chính sách `π_θ` được tham số hóa bởi `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

trong đó `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` là lợi tức chiết khấu từ bước `t`. Kỳ vọng lấy trên toàn bộ quỹ đạo (trajectory) `τ` được lấy mẫu từ `π_θ`.

**Chứng minh rất ngắn.** Vi phân `J(θ) = Σ_τ P(τ; θ) G(τ)` dưới dấu kỳ vọng. Dùng `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (thủ thuật đạo hàm log — log-derivative trick). Phân tích `log P(τ; θ) = Σ log π_θ(a_t | s_t) + các số hạng môi trường không phụ thuộc θ`. Các số hạng môi trường triệt tiêu. Hai dòng đại số là ra định lý.

**Các thủ thuật giảm phương sai (variance reduction).** REINFORCE nguyên bản có phương sai cực lớn — lợi tức nhiễu, `∇ log π` nhiễu, tích của chúng rất nhiễu. Hai cách sửa chuẩn:

1. **Trừ đường cơ sở.** Thay `G_t` bằng `G_t - b(s_t)` với mọi đường cơ sở (baseline) `b(s_t)` không phụ thuộc `a_t`. Vẫn không lệch vì `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Lựa chọn điển hình: `b(s_t) = V̂(s_t)` học bởi một critic → actor-critic (Bài 07).
2. **Reward-to-go.** Thay `Σ_t G_t · ∇ log π_θ(a_t | s_t)` bằng `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Chỉ lợi tức tương lai mới quan trọng cho một hành động — phần thưởng quá khứ chỉ đóng góp nhiễu trung bình không.

Gộp lại, bạn được:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

tức REINFORCE có đường cơ sở — tổ tiên trực tiếp của A2C (Bài 07) và PPO (Bài 08).

**Tham số hóa chính sách softmax.** Với hành động rời rạc, lựa chọn chuẩn:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

trong đó `f_θ` là mạng nơ-ron bất kỳ xuất ra điểm số cho mỗi hành động. Gradient có dạng gọn:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

tức điểm số của hành động đã chọn trừ đi kỳ vọng của nó dưới chính sách.

**Chính sách Gaussian cho hành động liên tục.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` có dạng đóng. Chỉ thế là đủ cho SAC ở Giai đoạn 9 · 07.

```figure
policy-gradient-landscape
```

## Tự tay làm

### Bước 1: mạng chính sách softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Dùng chính sách tuyến tính (một vector trọng số cho mỗi hành động) cho môi trường dạng bảng. Với Atari, hãy thay bằng CNN và giữ nguyên đầu ra softmax.

### Bước 2: lấy mẫu và log-xác suất

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Bước 3: rollout có ghi lại log-xác suất

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Bước 4: cập nhật REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)` (onehot của `a` trừ xác suất) là trái tim của gradient chính sách softmax. Hãy học cho thuộc xương.

### Bước 5: đường cơ sở

Trung bình chạy của `G` qua các episode gần đây đủ để giảm phương sai và chạy được GridWorld 4×4; cần ~500 episode để hội tụ. Nâng đường cơ sở lên `V̂(s)` học được là bạn có actor-critic.

## Cạm bẫy

- **Gradient bùng nổ.** Lợi tức có thể rất lớn. Luôn chuẩn hóa `G` về `~N(0, 1)` trên cả batch trước khi nhân với `∇ log π`.
- **Sụp đổ entropy.** Chính sách hội tụ về một hành động gần như tất định quá sớm, ngừng khám phá, rồi kẹt. Sửa: thêm thưởng entropy `β · H(π(·|s))` vào mục tiêu.
- **Phương sai cao.** REINFORCE nguyên bản cần hàng nghìn episode. Đường cơ sở từ critic (Bài 07) hoặc vùng tin cậy của TRPO/PPO (Bài 08) là cách sửa chuẩn.
- **Kém hiệu quả mẫu.** On-policy nghĩa là bạn vứt mọi transition sau một lần cập nhật. Hiệu chỉnh off-policy bằng importance sampling mang dữ liệu trở lại, đổi lấy phương sai (tỉ lệ trong PPO là trọng số IS đã bị cắt).
- **Gradient không dừng.** Cùng một gradient từ 100 episode trước dùng `π` cũ. Các phương pháp on-policy cập nhật sau vài rollout vì lý do này.
- **Gán công trạng.** Không có reward-to-go, phần thưởng quá khứ chỉ đóng góp nhiễu. Luôn dùng reward-to-go.

## Áp dụng

Năm 2026, REINFORCE hiếm khi được chạy trực tiếp nhưng công thức gradient của nó có mặt khắp nơi:

| Trường hợp dùng | Phương pháp suy ra |
|----------|---------------|
| Điều khiển liên tục | PPO / SAC với chính sách Gaussian |
| RLHF cho LLM | PPO với phạt KL, chạy trên chính sách mức token |
| Suy luận của LLM (DeepSeek) | GRPO — REINFORCE với đường cơ sở tương đối theo nhóm, không critic |
| Đa tác tử | REINFORCE critic tập trung (MADDPG, COMA) |
| Robot với hành động rời rạc | A2C, A3C, PPO |
| Chỉ có dữ liệu ưu tiên | DPO — REINFORCE viết lại thành loss hợp lý của ưu tiên, không lấy mẫu |

Khi bạn đọc `loss = -advantage * log_prob` trong một script huấn luyện năm 2026, đó chính là REINFORCE có đường cơ sở. Cả những bài báo (DPO, GRPO, RLOO) chỉ là các thủ thuật giảm phương sai đắp lên đúng một dòng đó.

## Đóng gói

Lưu thành `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Bài tập

1. **Dễ.** Cài REINFORCE trên GridWorld 4×4 với chính sách softmax tuyến tính. Huấn luyện 1.000 episode không có đường cơ sở. Vẽ đường cong học; đo phương sai (độ lệch chuẩn của lợi tức).
2. **Trung bình.** Thêm đường cơ sở trung bình chạy. Huấn luyện lại. So sánh hiệu quả mẫu và phương sai với lần chạy nguyên bản. Đường cơ sở giảm bao nhiêu bước cho tới khi hội tụ?
3. **Khó.** Thêm thưởng entropy `β · H(π)`. Quét `β ∈ {0, 0.01, 0.1, 1.0}`. Vẽ lợi tức cuối và entropy của chính sách. Điểm ngọt cho bài toán này nằm ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Policy gradient | "Huấn luyện chính sách trực tiếp" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; suy ra từ thủ thuật đạo hàm log. |
| REINFORCE | "Thuật toán PG nguyên bản" | Williams (1992); lợi tức Monte Carlo (Monte Carlo — ước lượng bằng cách lấy mẫu cả episode) nhân với gradient log-chính sách. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; làm cho gradient của kỳ vọng trở nên tính được. |
| Baseline | "Giảm phương sai" | Mọi `b(s)` bị trừ khỏi `G`; không lệch vì `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Chỉ lợi tức tương lai mới tính" | `G_t^{from t}` thay cho `G_0` đầy đủ; đúng và phương sai thấp hơn. |
| Entropy bonus | "Khuyến khích khám phá" | Số hạng `+β · H(π(\|s))` giữ chính sách không bị sụp. |
| On-policy | "Huấn luyện trên thứ vừa thấy" | Kỳ vọng gradient tính theo chính sách hiện tại — không thể dùng lại dữ liệu cũ trực tiếp. |
| Advantage | "Tốt hơn mức trung bình bao nhiêu" | `A(s, a) = G(s, a) - V(s)`; đại lượng có dấu mà REINFORCE-có-đường-cơ-sở nhân vào. |

## Đọc thêm

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — bài báo REINFORCE gốc.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — định lý gradient chính sách hiện đại với xấp xỉ hàm.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — cách trình bày trong sách giáo khoa.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — giải thích sư phạm rõ ràng kèm code PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — giảm phương sai và góc nhìn gradient tự nhiên nối REINFORCE với họ vùng tin cậy (TRPO, PPO).

---

[← Bài trước: Mạng Q sâu (Deep Q-Networks, DQN)](../../05-dqn/docs/vi.md) · [Bài sau: Actor-Critic →](../../07-actor-critic-a2c-a3c/docs/vi.md)
