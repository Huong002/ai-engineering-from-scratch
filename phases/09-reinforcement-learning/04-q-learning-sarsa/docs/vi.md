# Temporal Difference — Q-Learning & SARSA

> Monte Carlo đợi đến khi episode kết thúc. TD cập nhật sau mỗi bước bằng cách bootstrapping (dùng chính ước lượng hiện tại để dựng mục tiêu). Q-learning là off-policy (học ngoài chính sách) và lạc quan; SARSA là on-policy (học theo chính sách) và thận trọng. Cả hai chỉ là một dòng code. Cả hai là nền móng cho mọi phương pháp deep-RL trong giai đoạn này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 01 (MDP), Giai đoạn 9, Bài 02 (Quy hoạch động), Giai đoạn 9, Bài 03 (Monte Carlo)
**Time:** ~75 phút

## Vấn đề

Monte Carlo chạy được nhưng có hai đòi hỏi đắt. Nó cần episode kết thúc, và nó chỉ cập nhật sau khi đã có return cuối cùng. Nếu episode của bạn dài 1.000 bước, MC đợi 1.000 bước mới cập nhật được gì. Nó có phương sai cao, độ chệch thấp, và chậm trên thực tế.

Quy hoạch động có chân dung ngược lại — backup bootstrap với phương sai bằng 0 — nhưng đòi hỏi biết trước mô hình.

Temporal difference (sai khác thời gian, TD) chia đôi khoảng cách đó. Từ một transition `(s, a, r, s')`, dựng mục tiêu một bước `r + γ V(s')` rồi đẩy `V(s)` về phía đó. Không cần mô hình. Không cần episode trọn vẹn. Có độ chệch vì vế phải dùng `V` xấp xỉ, nhưng phương sai thấp hơn MC rất nhiều và cập nhật online ngay từ bước đầu.

Đây là bản lề mà toàn bộ RL hiện đại — DQN, A2C, PPO, SAC — xoay quanh. Phần còn lại của Giai đoạn 9 là các lớp xấp xỉ hàm và mẹo xây trên bản cập nhật TD một bước mà bạn sẽ viết trong bài này.

## Khái niệm

![Q-learning vs SARSA: max off-policy so với Q(s', a') on-policy](../assets/td.svg)

**Cập nhật TD(0) cho V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

Đại lượng trong ngoặc là TD error `δ = r + γ V(s') - V(s)`. Nó là bản online của `G_t - V(s_t)` trong MC. Hội tụ cần `α` thoả điều kiện Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) và mọi trạng thái được thăm vô hạn lần.

**Q-learning.** Một phương pháp TD off-policy dùng cho control:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

Phép `max` giả định chính sách *greedy* sẽ được theo từ `s'` trở đi, bất kể agent thực sự chọn hành động nào. Sự tách rời đó khiến Q-learning học `Q*` trong khi agent khám phá bằng epsilon-greedy (ε-tham lam, cân bằng exploration (khám phá) và exploitation (khai thác)). Mnih et al. (2015) biến nó thành deep Q-learning trên Atari (Bài 05).

**SARSA.** Một phương pháp TD on-policy:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Tên gọi chính là bộ tuple `(s, a, r, s', a')`. SARSA dùng hành động `a'` mà agent *thực sự* chọn tiếp theo, không phải `argmax` tham lam. Nó hội tụ về `Q^π` với bất kỳ `π` epsilon-greedy nào đang chạy, và trong giới hạn `ε → 0` thì thành `Q*`.

**Khác biệt ở cliff-walking.** Trên bài toán cliff-walking kinh điển (rơi xuống vực = phần thưởng -100), Q-learning học đường tối ưu men theo mép vực nhưng thỉnh thoảng vẫn ăn phạt trong lúc khám phá. SARSA học một đường an toàn hơn, cách mép vực một bước, vì nó tính cả nhiễu khám phá vào giá trị Q của mình. Khi huấn luyện đủ lâu, cả hai đều đạt tối ưu tại `ε → 0`. Trên thực tế điều này quan trọng: khi việc khám phá vẫn thực sự diễn ra lúc triển khai, hành vi của SARSA thận trọng hơn.

**Expected SARSA.** Thay `Q(s', a')` bằng kỳ vọng của nó dưới `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Phương sai thấp hơn SARSA (không lấy mẫu `a'`), cùng mục tiêu on-policy. Thường là lựa chọn mặc định trong các sách giáo trình hiện đại.

**n-step TD và TD(λ).** Nội suy giữa TD(0) và MC bằng cách đợi `n` bước rồi mới bootstrap. `n=1` là TD, `n=∞` là MC. TD(λ) lấy trung bình trên mọi `n` với trọng số hình học `(1-λ)λ^{n-1}`. Phần lớn deep-RL dùng `n` trong khoảng 3 đến 20.

```figure
qlearning-gridworld
```

## Tự tay làm

### Bước 1: SARSA trên chính sách ε-greedy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Tám dòng. Khác biệt *duy nhất* so với Q-learning nằm ở dòng target.

### Bước 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

Phép `max` tách mục tiêu khỏi hành vi. Chính một ký hiệu đó là ranh giới giữa on-policy và off-policy.

### Bước 3: đường cong học

Theo dõi return trung bình mỗi 100 episode. Q-learning hội tụ nhanh hơn trên GridWorld tất định đơn giản; SARSA thận trọng hơn trên cliff-walking. Trên GridWorld 4×4 trong `code/main.py`, cả hai đều gần tối ưu sau ~2.000 episode với `α=0.1, ε=0.1`.

### Bước 4: so sánh với chân lý DP

Chạy value iteration (Bài 02) để có `Q*`. Kiểm tra `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Một agent TD dạng bảng khoẻ mạnh sẽ nằm trong khoảng `~0.5` trên GridWorld 4×4 sau 10.000 episode.

## Cạm bẫy

- **Giá trị Q khởi tạo rất quan trọng.** Khởi tạo lạc quan (`Q = 0` với bài toán phần thưởng âm) khuyến khích khám phá. Khởi tạo bi quan có thể nhốt một chính sách tham lam mãi mãi.
- **Lịch α.** `α` hằng số là ổn cho bài toán không dừng. `α_n = 1/n` giảm dần đảm bảo hội tụ trên lý thuyết nhưng quá chậm trên thực tế — chốt `α` trong `[0.05, 0.3]` và theo dõi đường cong học.
- **Lịch ε.** Bắt đầu cao (`ε=1.0`), giảm dần về `ε=0.05`. "GLIE" (greedy in the limit with infinite exploration) là điều kiện hội tụ.
- **Chệch do max trong Q-learning.** Toán tử `max` bị chệch lên trên khi `Q` còn nhiễu. Dẫn tới ước lượng quá cao — Double Q-learning của Hasselt (được DDQN dùng trong Bài 05) sửa việc này bằng hai Q-table (bảng Q).
- **Episode không kết thúc.** TD có thể học mà không cần trạng thái kết thúc, nhưng bạn cần hoặc chặn số bước hoặc xử lý bootstrap đúng tại điểm chặn. Chuẩn mực: coi điểm chặn là không kết thúc, vẫn bootstrap tiếp.
- **Băm trạng thái.** Nếu trạng thái là tuple/tensor, hãy dùng khoá băm được (tuple, không phải list; tuple các số thực đã làm tròn, không dùng số thô).

## Áp dụng

Bức tranh TD năm 2026:

| Tác vụ | Phương pháp | Lý do |
|------|--------|--------|
| Môi trường bảng nhỏ | Q-learning | Học thẳng chính sách tối ưu. |
| On-policy, an toàn trọng yếu | SARSA / Expected SARSA | Thận trọng trong lúc khám phá. |
| Trạng thái nhiều chiều | DQN (Giai đoạn 9 · 05) | Hàm Q bằng mạng nơ-ron với replay và target net. |
| Hành động liên tục | SAC / TD3 (Giai đoạn 9 · 07) | Cập nhật TD trên mạng Q; mạng chính sách phát ra hành động. |
| RL cho LLM (dựa trên reward model) | PPO / GRPO (Giai đoạn 9 · 08, 12) | Actor-critic với advantage kiểu TD qua GAE. |
| Offline RL | CQL / IQL (Giai đoạn 9 · 08) | Q-learning với điều chuẩn thận trọng. |

Chín mươi phần trăm những gì bạn đọc về "RL" trong các bài báo năm 2026 chỉ là một biến thể nào đó của Q-learning hoặc SARSA. Hãy hiểu bản cập nhật dạng bảng đến mức gõ lại được trước khi đọc sâu hơn.

## Đóng gói

Lưu thành `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Bài tập

1. **Dễ.** Cài Q-learning và SARSA trên GridWorld 4×4. Vẽ đường cong học (return trung bình mỗi 100 episode) cho 2.000 episode. Bên nào hội tụ nhanh hơn?
2. **Trung bình.** Dựng môi trường cliff-walking (4×12, hàng cuối là vực với phần thưởng -100 và reset về điểm xuất phát). So sánh chính sách cuối của Q-learning và SARSA. Chụp lại đường đi của mỗi bên. Bên nào sát mép vực hơn?
3. **Khó.** Cài Double Q-learning. Trên GridWorld có phần thưởng nhiễu (thêm nhiễu Gaussian σ=5 vào phần thưởng mỗi bước), cho thấy Q-learning ước lượng quá cao `V*(0,0)` một lượng đáng kể trong khi Double Q-learning thì không.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| TD error | "Tín hiệu cập nhật" | `δ = r + γ V(s') - V(s)`, phần dư đã bootstrap. |
| TD(0) | "TD một bước" | Cập nhật sau mỗi transition, chỉ dùng ước lượng của trạng thái kế tiếp. |
| Q-learning | "RL off-policy nhập môn" | Cập nhật TD với `max` trên các hành động của trạng thái kế; học `Q*` bất kể chính sách hành vi. |
| SARSA | "Q-learning on-policy" | Cập nhật TD dùng hành động kế tiếp thực tế; học `Q^π` cho π epsilon-greedy hiện tại. |
| Expected SARSA | "SARSA phương sai thấp" | Thay `a'` lấy mẫu bằng kỳ vọng của nó dưới π. |
| GLIE | "Lịch khám phá đúng" | Greedy in the Limit with Infinite Exploration; cần cho sự hội tụ của Q-learning. |
| Bootstrapping | "Dùng ước lượng hiện tại trong mục tiêu" | Điều phân biệt TD với MC. Nguồn gốc của độ chệch nhưng giảm phương sai rất mạnh. |
| Maximization bias | "Q-learning ước lượng quá cao" | `max` trên các ước lượng nhiễu bị chệch lên trên; Double Q-learning sửa được. |

## Đọc thêm

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — bài báo gốc và chứng minh hội tụ.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — bản sửa cho maximization bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — động lực của expected SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — bài báo đặt ra cái tên SARSA (khi đó gọi là "modified connectionist Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — tổng quát hoá TD(0) thành TD(n), con đường từ Q-learning tới eligibility traces và về sau là GAE trong PPO.

---

[← Bài trước: Phương pháp Monte Carlo](../../03-monte-carlo-methods/docs/vi.md) · [Bài sau: Mạng Q sâu (Deep Q-Networks, DQN) →](../../05-dqn/docs/vi.md)
