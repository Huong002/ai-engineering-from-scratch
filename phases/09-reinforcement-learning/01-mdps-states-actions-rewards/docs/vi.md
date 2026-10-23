# MDP, Trạng thái, Hành động & Phần thưởng

> Một Markov decision process (quá trình quyết định Markov) gồm năm thứ: trạng thái, hành động, chuyển trạng thái, phần thưởng, hệ số chiết khấu. Mọi thứ trong RL — Q-learning, PPO, DPO, GRPO — đều tối ưu trên đúng hình dạng này. Học một lần, đọc phần còn lại của reinforcement learning mà không tốn thêm gì.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 1, Bài 06 (Xác suất & phân phối), Giai đoạn 2, Bài 01 (Phân loại ML)
**Time:** ~45 phút

## Vấn đề

Bạn đang viết một bot cờ vua. Hoặc một hệ thống lập kế hoạch tồn kho. Hoặc một agent giao dịch. Hoặc vòng lặp PPO huấn luyện một mô hình suy luận. Bốn lĩnh vực khác nhau, một sự thật bất ngờ: cả bốn đều quy về cùng một đối tượng toán học.

Học có giám sát cho bạn các cặp `(x, y)` và yêu cầu bạn khớp một hàm. Reinforcement learning không cho bạn nhãn — chỉ một dòng trạng thái, những hành động bạn đã làm, và một phần thưởng vô hướng. Nước đi đó có thắng ván cờ không? Quyết định nhập thêm hàng có tiết kiệm tiền không? Thương vụ đó có sinh lời không? Token mà LLM vừa sinh ra có dẫn tới phần thưởng cao hơn từ giám khảo không?

Bạn không thể học từ dòng dữ liệu này cho tới khi hình thức hoá nó. "Cái tôi thấy," "cái tôi làm," "cái xảy ra tiếp theo," "điều đó tốt đến đâu" — mỗi thứ phải trở thành một đối tượng mà bạn có thể lập luận. Sự hình thức hoá đó chính là một Markov decision process. Mọi thuật toán RL trong giai đoạn này, kể cả các vòng lặp RLHF và GRPO ở cuối, đều tối ưu trên hình dạng này.

## Khái niệm

![Quá trình quyết định Markov: trạng thái, hành động, chuyển trạng thái, phần thưởng, hệ số chiết khấu](../assets/mdp.svg)

**Năm đối tượng.**

- **States** (trạng thái) `S`. Mọi thứ agent cần để quyết định. Trong GridWorld, đó là ô lưới. Trong cờ vua, bàn cờ. Trong LLM, cửa sổ ngữ cảnh cộng thêm bộ nhớ.
- **Actions** (hành động) `A`. Các lựa chọn. Đi lên/xuống/trái/phải. Đi một nước. Phát ra một token.
- **Transitions** (chuyển trạng thái) `P(s' | s, a)`. Cho trạng thái `s` và hành động `a`, phân phối trên trạng thái kế tiếp. Tất định trong cờ vua, ngẫu nhiên trong bài toán tồn kho, gần như tất định trong giải mã LLM.
- **Rewards** (phần thưởng) `R(s, a, s')`. Tín hiệu vô hướng. Thắng = +1, thua = -1. Doanh thu trừ chi phí. Số hạng tỉ lệ log-likelihood trong GRPO.
- **Discount** (hệ số chiết khấu) `γ ∈ [0, 1)`. Phần thưởng tương lai được tính bao nhiêu so với hiện tại. `γ = 0.99` mua được chân trời ~100 bước; `γ = 0.9` mua được ~10.

**Thuộc tính Markov** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Tương lai chỉ phụ thuộc vào trạng thái hiện tại. Nếu không, cách biểu diễn trạng thái đã thiếu — đó không phải phương pháp sai, mà là trạng thái sai.

**Chính sách và lợi tức.** Một policy (chính sách) `π(a | s)` ánh xạ trạng thái sang phân phối hành động. Lợi tức `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` là tổng chiết khấu của phần thưởng tương lai. Giá trị `V^π(s) = E[G_t | s_t = s]` là lợi tức kỳ vọng khi bắt đầu từ `s` dưới chính sách `π`. Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` là lợi tức kỳ vọng khi bắt đầu bằng một hành động cụ thể. Mọi thuật toán RL đều ước lượng một trong hai đại lượng này, rồi cải thiện `π` tương ứng.

**Phương trình Bellman.** Các phương trình điểm bất động mà mọi thứ trong giai đoạn này dùng:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Chúng tách lợi tức kỳ vọng thành "phần thưởng của bước này" cộng "giá trị đã chiết khấu của nơi bạn đáp xuống." Đệ quy. Mọi thuật toán trong Giai đoạn 9 hoặc lặp phương trình này tới hội tụ (quy hoạch động), hoặc lấy mẫu từ nó (Monte Carlo), hoặc bootstrap nó một bước (temporal difference).

```figure
discount-horizon
```

## Tự tay làm

### Bước 1: một MDP tất định nhỏ xíu

Một GridWorld 4×4. Agent bắt đầu ở góc trên trái, ô kết thúc ở góc dưới phải, phần thưởng -1 mỗi bước, hành động `{up, down, left, right}`. Xem `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Năm dòng. Đó là toàn bộ môi trường. Chuyển trạng thái tất định, hình phạt mỗi bước không đổi, trạng thái kết thúc hấp thụ.

### Bước 2: chạy thử một chính sách

Một chính sách là hàm từ trạng thái sang phân phối hành động. Đơn giản nhất: ngẫu nhiên đều.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Chạy chính sách ngẫu nhiên 1000 lần. Lợi tức trung bình rơi vào khoảng -60 đến -80 với bàn 4×4 này. Lợi tức tối ưu là -6 (đường thẳng xuống phải). Thu hẹp khoảng cách đó là toàn bộ nội dung của Giai đoạn 9.

### Bước 3: tính chính xác `V^π` bằng phương trình Bellman

Với MDP nhỏ, phương trình Bellman là một hệ tuyến tính. Liệt kê các trạng thái, áp kỳ vọng, lặp cho tới khi giá trị không còn đổi nữa.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Đây là đánh giá chính sách lặp. Nó là thuật toán đầu tiên trong Sutton & Barto và là nền tảng lý thuyết của mọi phương pháp RL về sau.

### Bước 4: `γ` là siêu tham số mang ý nghĩa vật lý

Chân trời hiệu dụng xấp xỉ `1 / (1 - γ)`. `γ = 0.9` → 10 bước. `γ = 0.99` → 100 bước. `γ = 0.999` → 1000 bước.

Quá thấp thì agent hành động thiển cận. Quá cao thì việc gán công lao trở nên nhiễu, vì nhiều bước đầu cùng chia sẻ trách nhiệm cho phần thưởng ở tương lai xa. RLHF cho LLM thường dùng `γ = 1` vì episode (một lượt chạy trọn vẹn) ngắn và có giới hạn. Bài toán điều khiển dùng `0.95–0.99`. Game chiến lược chân trời dài dùng `0.999`.

## Cạm bẫy

- **Trạng thái không Markov.** Nếu bạn cần ba quan sát gần nhất để quyết định, thì "trạng thái" không chỉ là quan sát hiện tại. Cách sửa: xếp chồng khung hình (DQN trên Atari xếp 4 khung) hoặc dùng trạng thái hồi quy (LSTM/GRU trên các quan sát).
- **Phần thưởng thưa.** Phần thưởng chỉ có khi thắng khiến việc học gần như bất khả thi trong không gian trạng thái lớn. Hãy định hình phần thưởng (tín hiệu trung gian) hoặc bootstrap bằng bắt chước (Giai đoạn 9 · 09).
- **Reward hacking.** Tối ưu một phần thưởng đại diện thường sinh ra hành vi bệnh lý. Agent đua thuyền của OpenAI quay vòng thu thập vật phẩm mãi mãi thay vì về đích. Luôn định nghĩa phần thưởng từ kết quả mục tiêu, không phải từ đại lượng đại diện.
- **Chọn sai hệ số chiết khấu.** `γ = 1` trên bài toán chân trời vô hạn làm mọi giá trị thành vô hạn. Luôn chặn bằng chân trời hữu hạn hoặc `γ < 1`.
- **Độ lớn phần thưởng.** Phần thưởng {+100, -100} so với {+1, -1} cho chính sách tối ưu giống hệt nhau nhưng độ lớn gradient khác xa. Chuẩn hoá về khoảng `[-1, 1]` trước khi đưa vào PPO/DQN.

## Áp dụng

Bộ công cụ của 2026 quy mọi pipeline RL về một MDP trước khi chạm vào code:

| Tình huống | Trạng thái | Hành động | Phần thưởng | γ |
|-----------|-------|--------|---|-------|
| Điều khiển (đi lại, thao tác) | Góc khớp + vận tốc | Mô-men liên tục | Định hình theo tác vụ | 0.99 |
| Game (cờ vua, Go, poker) | Bàn cờ + lịch sử | Nước đi hợp lệ | Thắng=+1 / thua=-1 | 1.0 (hữu hạn) |
| Tồn kho / định giá | Tồn kho + nhu cầu | Số lượng đặt hàng | Doanh thu - chi phí | 0.95 |
| RLHF cho LLM | Token ngữ cảnh | Token kế tiếp | Điểm của mô hình thưởng ở cuối | 1.0 (episode ~200 token) |
| GRPO cho suy luận | Prompt + phản hồi một phần | Token kế tiếp | Bộ kiểm 0/1 ở cuối | 1.0 |

Viết năm bộ tuple trước khi viết bất kỳ vòng huấn luyện nào. Phần lớn báo lỗi "RL không chạy" truy về một công thức MDP đã hỏng ngay trên giấy.

## Đóng gói

Lưu thành `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Bài tập

1. **Dễ.** Cài đặt GridWorld 4×4 và rollout chính sách ngẫu nhiên trong `code/main.py`. Chạy 10.000 episode. Báo cáo trung bình và độ lệch chuẩn của lợi tức. So sánh với lợi tức tối ưu (-6).
2. **Trung bình.** Chạy `policy_evaluation` với `γ ∈ {0.5, 0.9, 0.99}` cho chính sách ngẫu nhiên đều. In `V` dưới dạng lưới 4×4 cho từng trường hợp. Giải thích vì sao giá trị các trạng thái gần ô kết thúc tăng nhanh hơn khi `γ` lớn hơn.
3. **Khó.** Làm GridWorld thành ngẫu nhiên: mỗi hành động trượt sang hướng kề bên với xác suất `p = 0.1`. Đánh giá lại chính sách ngẫu nhiên đều. `V[start]` tốt lên hay xấu đi? Vì sao?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| MDP | "Thiết lập của reinforcement learning" | Tuple `(S, A, P, R, γ)` thoả thuộc tính Markov. |
| State | "Cái agent nhìn thấy" | Thống kê đủ cho động lực tương lai dưới lớp chính sách đã chọn. |
| Policy | "Hành vi của agent" | Phân phối có điều kiện `π(a \| s)` hoặc ánh xạ tất định `s → a`. |
| Return | "Tổng phần thưởng" | Tổng đã chiết khấu `Σ γ^t r_t` tính từ bước hiện tại. |
| Value | "Một trạng thái tốt đến đâu" | Lợi tức kỳ vọng dưới `π` khi bắt đầu từ `s`. |
| Q-value | "Một hành động tốt đến đâu" | Lợi tức kỳ vọng dưới `π` khi bắt đầu từ `s` với hành động đầu tiên là `a`. |
| Bellman equation | "Đệ quy quy hoạch động" | Phân rã điểm bất động của value / Q thành phần thưởng một bước cộng giá trị kế tiếp đã chiết khấu. |
| Discount `γ` | "Tương lai so với hiện tại" | Trọng số hình học đặt lên phần thưởng tương lai xa; chân trời hiệu dụng `~1/(1-γ)`. |

## Đọc thêm

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — cuốn sách giáo khoa. Ch. 3 trình bày MDP và phương trình Bellman; Ch. 1 tạo động lực cho giả thuyết phần thưởng, nền tảng của mọi bài học sau.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — nguồn gốc của phương trình Bellman.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — bản nhập môn MDP súc tích từ góc nhìn deep RL.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — tài liệu tham chiếu của ngành vận trù về MDP và các phương pháp giải chính xác.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://cs.brown.edu/media/filer_public/d1/a6/d1a6f66a-289a-4b81-9596-417114843489/littman.pdf) — cách dẫn ra MDP sạch nhất như một chuyên biệt hoá của quy hoạch động.

---

[Bài sau: Quy hoạch động →](../../02-dynamic-programming/docs/vi.md)
