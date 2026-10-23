# Quy hoạch động — Lặp chính sách & Lặp giá trị

> Dynamic programming (quy hoạch động) là RL chơi gian. Bạn đã biết hàm chuyển trạng thái và hàm phần thưởng; bạn chỉ việc lặp phương trình Bellman (Bellman equation) cho tới khi `V` hoặc `π` đứng yên. Đây là chuẩn mực mà mọi phương pháp lấy mẫu cố gắng tiến tới.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 01 (MDP)
**Time:** ~75 phút

## Vấn đề

Bạn có một MDP đã biết mô hình: bạn có thể truy vấn `P(s' | s, a)` và `R(s, a, s')` với mọi cặp trạng thái–hành động. Một người quản lý tồn kho biết phân phối nhu cầu. Một ván cờ bàn có chuyển trạng thái tất định. Một gridworld chỉ là bốn dòng Python. Bạn có một *mô hình*.

RL không mô hình (Q-learning, PPO, REINFORCE) ra đời cho trường hợp bạn không có mô hình — bạn chỉ có thể lấy mẫu từ môi trường. Nhưng khi đã có mô hình, lại có những phương pháp nhanh hơn, tốt hơn: quy hoạch động. Bellman thiết kế chúng năm 1957. Đến nay chúng vẫn định nghĩa tính đúng đắn: khi người ta nói "chính sách tối ưu cho MDP này", họ muốn nói chính sách mà DP sẽ trả về.

Năm 2026, bạn cần chúng vì ba lý do. Thứ nhất, mọi môi trường dạng bảng trong nghiên cứu RL (GridWorld, FrozenLake, CliffWalking) đều được giải bằng DP để tạo ra chính sách chuẩn vàng. Thứ hai, giá trị chính xác cho phép bạn *gỡ lỗi* các phương pháp lấy mẫu: nếu ước lượng `V*(s_0)` của Q-learning lệch 30% so với đáp án DP, thì Q-learning của bạn có bug. Thứ ba, các phương pháp RL ngoại tuyến và lập kế hoạch hiện đại (MCTS, tìm kiếm của AlphaZero, RL dựa trên mô hình ở Giai đoạn 9 · 10) đều lặp một backup (lan giá trị) Bellman trên mô hình đã học hoặc mô hình cho sẵn.

## Khái niệm

![Lặp chính sách và lặp giá trị, đặt cạnh nhau](../assets/dp.svg)

**Hai thuật toán, cả hai đều là lặp điểm bất động trên Bellman.**

**Lặp chính sách (policy iteration).** Luân phiên hai bước cho tới khi chính sách không đổi nữa.

1. *Đánh giá:* với chính sách `π`, tính `V^π` bằng cách lặp lại `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` cho tới khi hội tụ (convergence).
2. *Cải thiện:* với `V^π`, cho `π` tham lam theo `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Hội tụ được bảo đảm vì (a) mỗi bước cải thiện hoặc giữ `π` nguyên hoặc làm `V^π` tăng thật sự ở một trạng thái nào đó, (b) không gian các chính sách tất định là hữu hạn. Thường chỉ cần ~5–20 vòng lặp ngoài, kể cả với không gian trạng thái lớn.

**Lặp giá trị (value iteration).** Gộp bước đánh giá và bước cải thiện vào một lượt quét. Áp dụng phương trình Bellman *tối ưu*:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Lặp cho tới khi `max_s |V_{new}(s) - V(s)| < ε`. Ở cuối, trích xuất chính sách bằng cách lấy hành động tham lam. Mỗi vòng lặp nhanh hơn hẳn — không có vòng đánh giá bên trong — nhưng thường cần nhiều vòng hơn để hội tụ.

**Lặp chính sách tổng quát (GPI).** Cách nhìn thống nhất. Hàm giá trị (value function) và chính sách khoá chặt vào nhau trong một vòng cải thiện hai chiều; bất kỳ phương pháp nào đẩy cả hai tới chỗ nhất quán với nhau (lặp giá trị bất đồng bộ, lặp chính sách có sửa đổi, Q-learning, actor-critic, PPO) đều là một trường hợp của GPI.

**Vì sao `γ < 1` quan trọng.** Toán tử Bellman là một ánh xạ co hệ số `γ` theo chuẩn sup: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Tính co kéo theo điểm bất động duy nhất và hội tụ hình học. Bỏ `γ < 1` là mất bảo đảm — khi đó bạn cần horizon hữu hạn hoặc một trạng thái kết thúc hấp thụ.

```figure
value-iteration-gamma
```

## Tự tay làm

### Bước 1: dựng mô hình MDP GridWorld

Dùng đúng GridWorld 4×4 của Bài 01. Ta thêm một biến thể ngẫu nhiên: với xác suất `0.1`, tác tử trượt sang một hướng vuông góc ngẫu nhiên.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` trả về danh sách các bộ `(s', r, p)`. Toàn bộ mô hình chỉ có vậy.

### Bước 2: đánh giá chính sách

Cho chính sách `π(s) = {action: prob}`, lặp phương trình Bellman cho tới khi `V` đứng yên:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Bước 3: cải thiện chính sách

Thay `π` bằng chính sách tham lam theo `V`. Nếu `π` không đổi thì trả về — ta đã ở điểm tối ưu.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Bước 4: ghép chúng lại

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Trên lưới 4×4, thường hội tụ sau 4–6 vòng ngoài. Cho ra `V*(0,0) ≈ -6` và một chính sách giảm hẳn số bước đi.

### Bước 5: lặp giá trị (bản một vòng lặp)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Cùng một điểm bất động, ít dòng code hơn.

## Cạm bẫy

- **Quên xử lý trạng thái kết thúc.** Nếu bạn áp Bellman lên một trạng thái hấp thụ, nó vẫn chọn ra một "hành động tốt nhất" mà chẳng thay đổi gì. Hãy chặn bằng `if s == terminal: V[s] = 0`.
- **Hội tụ theo chuẩn sup so với L2.** Dùng `max |V_new - V|`, đừng dùng trung bình. Bảo đảm lý thuyết nằm ở chuẩn sup.
- **Cập nhật tại chỗ so với cập nhật đồng bộ.** Cập nhật `V[s]` tại chỗ (Gauss-Seidel) hội tụ nhanh hơn so với dùng một dict `V_new` riêng (Jacobi). Code chạy thật dùng cách tại chỗ.
- **Chính sách hoà nhau.** Nếu hai hành động có Q-value bằng nhau, `argmax` có thể phá hoà theo cách khác nhau ở mỗi vòng, khiến phép kiểm tra "chính sách ổn định" dao động. Hãy dùng cách phá hoà ổn định (hành động đầu tiên theo thứ tự cố định).
- **Bùng nổ không gian trạng thái.** DP tốn `O(|S| · |A|)` mỗi lượt quét. Dùng được tới ~10⁷ trạng thái. Vượt mức đó, bạn cần xấp xỉ hàm (Giai đoạn 9 · 05 trở đi).

## Áp dụng

Năm 2026, DP là chuẩn đối chiếu về tính đúng đắn và là vòng lặp bên trong của các bộ lập kế hoạch:

| Trường hợp dùng | Phương pháp |
|----------|--------|
| Giải chính xác một MDP dạng bảng nhỏ | Lặp giá trị (đơn giản hơn) hoặc lặp chính sách (ít vòng ngoài hơn) |
| Kiểm chứng một bản cài Q-learning / PPO | So sánh với V* tối ưu từ DP trên một môi trường đồ chơi |
| RL dựa trên mô hình (Giai đoạn 9 · 10) | Backup Bellman trên mô hình chuyển trạng thái đã học |
| Lập kế hoạch trong AlphaZero / MuZero | Monte Carlo Tree Search = backup Bellman bất đồng bộ |
| RL ngoại tuyến (CQL, IQL) | Q-iteration thận trọng — DP kèm khoản phạt cho hành động OOD |

Mỗi khi ai đó nói "hàm giá trị tối ưu", họ muốn nói "điểm bất động của DP". Khi thấy `V*` hoặc `Q*` trong một bài báo, hãy hình dung vòng lặp này.

## Đóng gói

Lưu thành `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Bài tập

1. **Dễ.** Chạy lặp giá trị trên GridWorld 4×4 với `γ ∈ {0.9, 0.99}`. Cần bao nhiêu lượt quét cho tới khi `max |ΔV| < 1e-6`? In `V*` dưới dạng lưới 4×4.
2. **Trung bình.** So sánh lặp chính sách với lặp giá trị trên GridWorld *ngẫu nhiên* (xác suất trượt `0.1`). Đếm: số lượt quét, thời gian thực, `V*(0,0)` cuối cùng. Cách nào hội tụ nhanh hơn về số vòng lặp? Về thời gian thực?
3. **Khó.** Xây bản lặp chính sách có sửa đổi: ở bước đánh giá chỉ chạy `k` lượt quét thay vì chạy tới hội tụ. Vẽ sai số của `V*(0,0)` theo `k` với `k ∈ {1, 2, 5, 10, 50}`. Đường cong đó cho bạn biết gì về đánh đổi giữa đánh giá và cải thiện?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Policy iteration | "thuật toán DP" | Luân phiên đánh giá (`V^π`) và cải thiện (`π` tham lam theo `V^π`) cho tới khi chính sách không đổi nữa. |
| Value iteration | "DP nhanh hơn" | Backup tối ưu Bellman áp trong một lượt quét; hội tụ về `V*` theo cấp số nhân. |
| Bellman operator | "Cái đệ quy đó" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; một ánh xạ co hệ số `γ` theo chuẩn sup. |
| Contraction | "Lý do DP hội tụ" | Mọi toán tử `T` có `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` đều có điểm bất động duy nhất. |
| GPI | "Mọi thứ đều là DP" | Generalized Policy Iteration: mọi phương pháp đẩy `V` và `π` tới chỗ nhất quán với nhau. |
| Synchronous update | "Kiểu Jacobi" | Dùng `V` cũ suốt một lượt quét; phân tích rạch ròi nhưng chậm hơn. |
| In-place update | "Kiểu Gauss-Seidel" | Dùng `V` ngay khi nó đang được cập nhật; trên thực tế hội tụ nhanh hơn. |

## Đọc thêm

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — cách trình bày chuẩn mực về lặp chính sách và lặp giá trị.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook_athena.html) — xử lý chặt chẽ các lập luận ánh xạ co.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — lặp chính sách có sửa đổi và phân tích hội tụ của nó.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — bài báo gốc về lặp chính sách.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — cầu nối từ DP sang DP xấp xỉ / deep RL mà mọi bài học sau đó đều dùng.

---

[← Bài trước: MDP, Trạng thái, Hành động & Phần thưởng](../../01-mdps-states-actions-rewards/docs/vi.md) · [Bài sau: Phương pháp Monte Carlo →](../../03-monte-carlo-methods/docs/vi.md)
