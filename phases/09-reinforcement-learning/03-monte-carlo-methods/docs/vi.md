# Phương pháp Monte Carlo — Học từ những episode trọn vẹn

> Quy hoạch động cần một mô hình. Monte Carlo không cần gì ngoài các episode (một lượt chơi trọn vẹn). Chạy chính sách, quan sát return (lợi tức), rồi lấy trung bình. Ý tưởng đơn giản nhất trong RL — và cũng là ý tưởng mở khoá mọi thứ về sau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 01 (MDP), Giai đoạn 9, Bài 02 (Quy hoạch động)
**Time:** ~75 phút

## Vấn đề

Quy hoạch động thì thanh lịch, nhưng nó giả định rằng bạn truy vấn được `P(s' | s, a)` cho mọi trạng thái và hành động. Gần như không gì trong thế giới thực hoạt động theo cách đó. Một robot không thể tính bằng giải tích phân phối trên các pixel camera sau khi một khớp chịu mô-men xoắn. Một thuật toán định giá không thể lấy tích phân trên mọi phản ứng có thể có của khách hàng. Một LLM không thể liệt kê mọi phần tiếp nối khả dĩ sau một token.

Bạn cần một phương pháp chỉ đòi hỏi khả năng *sampling* (lấy mẫu) từ môi trường. Chạy chính sách. Thu được một quỹ đạo `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Dùng nó để ước lượng giá trị. Đó chính là Monte Carlo (ước lượng dựa trên lấy mẫu).

Bước chuyển từ DP sang MC mang ý nghĩa triết học: ta đi từ *mô hình đã biết + lan truyền chính xác* sang *rollout lấy mẫu + lợi tức lấy trung bình*. Variance (phương sai) tăng vọt, nhưng phạm vi áp dụng bùng nổ. Mọi thuật toán RL sau bài này — TD, Q-learning, REINFORCE, PPO, GRPO — về bản chất đều là một bộ ước lượng Monte Carlo, đôi khi có thêm bootstrapping chồng lên trên.

## Khái niệm

![Monte Carlo: rollout, tính lợi tức, lấy trung bình; first-visit so với every-visit](../assets/monte-carlo.svg)

**Ý tưởng cốt lõi, gói trong một dòng:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` trong đó `G^{(i)}(s)` là các lợi tức quan sát được sau những lần ghé `s` dưới chính sách `π`.

**First-visit (lần ghé thăm đầu tiên) vs every-visit (mọi lần ghé thăm) MC.** Với một episode ghé trạng thái `s` nhiều lần, first-visit MC chỉ tính lợi tức của lần ghé đầu; every-visit MC tính mọi lần ghé. Cả hai đều không chệch khi lấy giới hạn. First-visit dễ phân tích hơn (mẫu iid). Every-visit dùng nhiều dữ liệu hơn trên mỗi episode và thường hội tụ nhanh hơn trong thực tế.

**Trung bình tích luỹ.** Thay vì lưu tất cả lợi tức, hãy cập nhật trung bình chạy:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Viết lại: `V_new = V_old + α · (target - V_old)` với `α = 1/n`. Đổi `1/n` thành step-size hằng `α ∈ (0, 1)` là bạn có một bộ ước lượng MC không dừng, bám theo thay đổi của `π`. Chính bước đó là toàn bộ cú nhảy từ MC sang TD rồi sang mọi thuật toán RL hiện đại.

**Giờ thì khám phá trở thành vấn đề.** DP chạm tới mọi trạng thái bằng cách liệt kê. MC chỉ thấy những trạng thái mà chính sách ghé qua. Nếu `π` là tất định, cả những vùng của không gian trạng thái sẽ không bao giờ được lấy mẫu, và ước lượng giá trị của chúng mãi mãi bằng 0. Ba cách khắc phục, theo thứ tự lịch sử:

1. **Exploring starts.** Bắt đầu mỗi episode từ một cặp (s, a) ngẫu nhiên. Đảm bảo độ phủ; không thực tế (bạn không thể "reset" một robot về một trạng thái tuỳ ý).
2. **ε-greedy.** Hành động tham lam theo Q hiện tại, nhưng với xác suất `ε` chọn một hành động ngẫu nhiên. Mọi cặp trạng thái-hành động đều được lấy mẫu khi số episode tiến ra vô hạn.
3. **Off-policy MC.** Thu dữ liệu theo chính sách hành vi `μ`, học về chính sách đích `π` qua importance sampling (lấy mẫu quan trọng). Phương sai cao, nhưng đây là cầu nối tới các phương pháp dùng replay buffer như DQN.

**Monte Carlo Control.** Đánh giá → cải thiện → đánh giá, y như lặp chính sách, nhưng bước đánh giá dựa trên lấy mẫu:

1. Chạy `π`, thu được một episode.
2. Cập nhật `Q(s, a)` từ các lợi tức quan sát được.
3. Đặt `π` thành ε-greedy theo `Q`.
4. Lặp lại.

Hội tụ về `Q*` và `π*` với xác suất 1 dưới những điều kiện nhẹ (mọi cặp được ghé vô hạn lần, `α` thoả điều kiện Robbins-Monro).

```figure
epsilon-greedy
```

## Tự tay làm

### Bước 1: rollout → danh sách các (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Không cần mô hình, chỉ cần `env.reset()` và `env.step(s, a)`. Giao diện giống môi trường gym nhưng đã lược bớt.

### Bước 2: tính lợi tức (duyệt ngược)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Một lượt duyệt, `O(T)`. Công thức truy hồi ngược `G_t = r_{t+1} + γ G_{t+1}` giúp khỏi phải cộng lại từ đầu.

### Bước 3: đánh giá MC first-visit

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Ba dòng làm hết việc: đánh dấu trạng thái đã gặp ở lần ghé đầu, tăng bộ đếm, cập nhật trung bình chạy.

### Bước 4: MC control ε-greedy (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Bước 5: so sánh với chuẩn vàng DP

Ước lượng MC của bạn về `V^π` phải khớp với kết quả DP ở Bài 02 khi số episode → ∞. Trên thực tế: 50,000 episode trên GridWorld 4×4 cho bạn sai lệch trong khoảng `~0.1` so với đáp án DP.

## Cạm bẫy

- **Episode vô hạn.** MC đòi hỏi episode phải *kết thúc*. Nếu chính sách của bạn có thể lặp mãi, hãy đặt trần `max_steps` và coi việc chạm trần là thất bại ngầm. GridWorld với chính sách ngẫu nhiên thường xuyên hết bước — đó là bình thường, chỉ cần đếm cho đúng.
- **Phương sai.** MC dùng toàn bộ lợi tức. Với episode dài, phương sai rất lớn — một phần thưởng xui xẻo ở cuối làm `V(s_0)` dịch đúng bằng lượng đó. Các phương pháp TD (Bài 04) cắt bớt chuyện này nhờ bootstrapping.
- **Độ phủ trạng thái.** MC tham lam trên một Q mới, khi có hoà, sẽ chỉ thử đúng một hành động. Bạn *buộc phải* khám phá (ε-greedy, exploring starts, UCB).
- **Chính sách không dừng.** Nếu `π` thay đổi (như trong MC control), những lợi tức cũ đến từ một chính sách khác. MC với α hằng xử lý được; MC lấy trung bình mẫu thì không.
- **Importance sampling off-policy.** Các trọng số `π(a|s)/μ(a|s)` nhân dồn dọc quỹ đạo. Phương sai nổ theo độ dài horizon. Hãy chặn bằng IS có trọng số theo từng quyết định hoặc chuyển sang TD.

## Áp dụng

Vai trò của Monte Carlo trong năm 2026:

| Trường hợp dùng | Vì sao dùng MC |
|----------|--------|
| Trò chơi ngắn hạn (blackjack, poker) | Episode kết thúc tự nhiên; lợi tức sạch. |
| Đánh giá offline một chính sách đã ghi log | Lấy trung bình lợi tức chiết khấu trên các quỹ đạo đã lưu. |
| Monte Carlo Tree Search (AlphaZero) | Các rollout MC từ lá của cây dẫn dắt việc chọn. |
| Đánh giá RL cho LLM | Tính phần thưởng trung bình trên các completion được lấy mẫu cho một chính sách. |
| Ước lượng baseline trong PPO | Mục tiêu advantage `A_t = G_t - V(s_t)` dùng `G_t` của MC. |
| Dạy RL | Thuật toán đơn giản nhất mà vẫn chạy được — bỏ bootstrapping để thấy phần cốt lõi. |

Các thuật toán deep-RL hiện đại (PPO, SAC) nội suy giữa MC thuần (toàn bộ lợi tức) và TD thuần (bootstrap một bước) qua lợi tức `n` bước hoặc GAE. Cả hai đầu mút đều là hiện thân của cùng một bộ ước lượng.

## Đóng gói

Lưu thành `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Bài tập

1. **Dễ.** Cài đặt đánh giá MC first-visit cho chính sách ngẫu nhiên đều trên GridWorld 4×4. Chạy 10,000 episode. Vẽ `V(0,0)` theo số episode và so với đáp án DP.
2. **Trung bình.** Cài đặt MC control ε-greedy với `ε ∈ {0.01, 0.1, 0.3}`. So sánh lợi tức trung bình sau 20,000 episode. Đường cong trông thế nào? Đánh đổi bias-variance nằm ở đâu?
3. **Khó.** Cài đặt MC *off-policy* với importance sampling: thu dữ liệu theo chính sách ngẫu nhiên đều `μ`, ước lượng `V^π` cho chính sách tối ưu tất định `π`. So sánh IS thường vs IS theo từng quyết định vs IS có trọng số. Cái nào có phương sai thấp nhất?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Monte Carlo | "Lấy mẫu ngẫu nhiên" | Ước lượng kỳ vọng bằng cách lấy trung bình trên các mẫu iid từ phân phối. |
| Return `G_t` | "Phần thưởng tương lai" | Tổng phần thưởng chiết khấu từ bước `t` tới cuối episode: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Mỗi trạng thái đếm một lần" | Chỉ lần ghé đầu trong một episode đóng góp vào ước lượng giá trị. |
| Every-visit MC | "Dùng mọi lần ghé" | Mọi lần ghé đều đóng góp; hơi chệch nhưng hiệu quả mẫu hơn. |
| ε-greedy | "Nhiễu khám phá" | Chọn hành động tham lam với xác suất `1-ε`; chọn ngẫu nhiên với xác suất `ε`. |
| Importance sampling | "Sửa lại vì lấy mẫu từ sai phân phối" | Đánh trọng số lại lợi tức bằng tích `π(a\|s)/μ(a\|s)` để ước lượng `V^π` từ dữ liệu của `μ`. |
| On-policy | "Học từ dữ liệu của chính mình" | Chính sách đích = chính sách hành vi. MC nguyên bản, PPO, SARSA. |
| Off-policy | "Học từ dữ liệu của người khác" | Chính sách đích ≠ chính sách hành vi. MC có importance sampling, Q-learning, DQN. |

## Đọc thêm

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — tài liệu kinh điển về chủ đề này.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — phân tích first-visit so với every-visit.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — MC off-policy và kiểm soát phương sai.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — các bộ ước lượng IS phương sai thấp hiện đại.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — minh chứng thực nghiệm quy mô lớn đầu tiên cho thấy tự chơi bằng MC/TD hội tụ tới trình độ siêu phàm; tiền thân về ý tưởng của mọi bài trong nửa sau giai đoạn này.

---

[← Bài trước: Quy hoạch động](../../02-dynamic-programming/docs/vi.md) · [Bài sau: Temporal Difference →](../../04-q-learning-sarsa/docs/vi.md)
