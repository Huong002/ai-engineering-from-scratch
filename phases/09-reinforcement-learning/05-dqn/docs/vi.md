# Mạng Q sâu (Deep Q-Networks, DQN)

> Năm 2013: Mnih huấn luyện một mạng Q-learning trên pixel thô, đánh bại mọi tác nhân RL cổ điển trên bảy trò Atari. Năm 2015: mở rộng ra 49 trò, công bố trên Nature, khơi mào kỷ nguyên deep-RL. DQN là Q-learning cộng thêm ba mẹo giúp xấp xỉ hàm trở nên ổn định.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 3, Bài 03 (Lan truyền ngược), Giai đoạn 9, Bài 04 (Q-learning, SARSA)
**Time:** ~75 phút

## Vấn đề

Q-learning dạng bảng cần một giá trị Q riêng cho từng cặp (state, action). Một bàn cờ có khoảng 10⁴³ trạng thái. Một khung hình Atari là 210×160×3 = 100,800 đặc trưng. RL dạng bảng chết ngay ở mức vài nghìn trạng thái, chứ đừng nói tới hàng tỷ.

Cách khắc phục nhìn lại thì hiển nhiên: thay bảng Q bằng một mạng nơ-ron, `Q(s, a; θ)`. Nhưng cái hiển nhiên khi nhìn lại đã mất hàng chục năm. Xấp xỉ hàm một cách ngây thơ với Q-learning sẽ phân kỳ dưới "bộ ba chết người" — xấp xỉ hàm + bootstrapping (khởi tạo từ chính ước lượng) + học off-policy. Mnih et al. (2013, 2015) đã chỉ ra ba mẹo kỹ thuật giúp việc học ổn định:

1. **Experience replay** làm mất tương quan giữa các bước chuyển.
2. **Target network** (mạng đích) đóng băng mục tiêu bootstrap.
3. **Reward clipping** chuẩn hoá độ lớn gradient.

DQN trên Atari là lần đầu tiên một kiến trúc duy nhất với một bộ siêu tham số duy nhất giải được hàng chục bài toán điều khiển từ pixel thô. Mọi thứ "deep-RL" xây từ đó đến nay — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — đều chồng lên nền ba mẹo này.

## Khái niệm

![Vòng lặp huấn luyện DQN: môi trường, bộ đệm phát lại, mạng online, mạng đích, mất mát TD Bellman](../assets/dqn.svg)

**Mục tiêu.** DQN cực tiểu hoá loss (mất mát) TD một bước trên một hàm Q dạng nơ-ron:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = mạng online, cập nhật mỗi bước bằng gradient descent. `θ^-` = mạng đích, được sao chép định kỳ từ `θ` (khoảng mỗi 10,000 bước). `D` = bộ đệm phát lại chứa các bước chuyển trong quá khứ.

**Ba mẹo, theo thứ tự quan trọng:**

**Experience replay.** Một ring buffer chứa `~10⁶` bước chuyển. Mỗi bước huấn luyện lấy mẫu ngẫu nhiên đều một minibatch. Điều này phá vỡ tương quan thời gian (các khung hình liên tiếp gần như giống nhau), cho phép mạng học nhiều lần từ những bước chuyển hiếm khi có thưởng, và làm mất tương quan giữa các cập nhật gradient liên tiếp. Không có nó, TD on-policy với mạng nơ-ron sẽ phân kỳ trên Atari.

**Target network.** Dùng cùng một mạng `Q(·; θ)` ở cả hai vế của phương trình Bellman khiến mục tiêu trở thành moving target (mục tiêu di động) sau mỗi lần cập nhật — "đuổi theo đuôi của chính mình". Cách khắc phục: giữ một mạng thứ hai `Q(·; θ^-)` với trọng số đóng băng. Cứ mỗi `C` bước, sao chép `θ → θ^-`. Điều này ổn định mục tiêu hồi quy trong hàng nghìn bước gradient liên tiếp. Cập nhật mềm `θ^- ← τ θ + (1-τ) θ^-` (dùng trong DDPG, SAC) là một biến thể mượt hơn.

**Reward clipping.** Độ lớn phần thưởng trong Atari dao động từ 1 đến hơn 1000. Cắt về `{-1, 0, +1}` ngăn một trò chơi duy nhất chi phối gradient. Sai khi độ lớn phần thưởng có ý nghĩa; ổn với Atari nơi chỉ dấu là quan trọng.

**Double DQN.** Hasselt (2016) sửa độ chệch tối đa hoá: dùng mạng online để *chọn* hành động, mạng đích để *đánh giá* nó.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Thay thế trực tiếp, luôn tốt hơn. Dùng nó mặc định.

**Các cải tiến khác (Rainbow, 2017):** prioritized replay (lấy mẫu nhiều hơn các bước chuyển có TD-error cao), kiến trúc dueling (tách riêng `V(s)` và các đầu advantage), noisy networks (khám phá được học), n-step returns, distributional Q (C51/QR-DQN), multi-step bootstrapping. Mỗi cái thêm vài phần trăm; lợi ích gần như cộng dồn.

```figure
f3-dqn-stability
```

## Tự tay làm

Code ở đây chỉ dùng stdlib, không dùng numpy — ta dùng một MLP một lớp ẩn tự viết trên một GridWorld liên tục tí hon, nên mỗi bước huấn luyện chạy trong vài micro giây. Ở quy mô lớn, thuật toán giống hệt DQN trên Atari.

### Bước 1: bộ đệm phát lại

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Dung lượng ~50,000 cho Atari; 5,000 là đủ cho môi trường đồ chơi của ta.

### Bước 2: một Q-network tí hon (MLP tự viết)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Lan truyền xuôi: linear → ReLU → linear. Toàn bộ mạng chỉ có thế.

### Bước 3: cập nhật DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Cấu trúc giống Q-learning ở Bài 04 với hai khác biệt: (a) ta lan truyền ngược qua một `Q(·; θ)` khả vi thay vì tra bảng, (b) mục tiêu dùng `Q(·; θ^-)`.

### Bước 4: vòng lặp ngoài

Với mỗi episode: hành động theo ε-greedy trên `Q(·; θ)`, đẩy các bước chuyển vào bộ đệm, lấy mẫu một minibatch, thực hiện một bước gradient, đồng bộ định kỳ `θ^- ← θ`. Mẫu hình:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Trên GridWorld tí hon của ta với state one-hot 16 chiều, tác nhân học được chính sách gần tối ưu trong khoảng 500 episode. Trên Atari, hãy mở rộng lên 200M khung hình và thêm bộ trích đặc trưng CNN.

## Cạm bẫy

- **Bộ ba chết người.** Xấp xỉ hàm + off-policy + bootstrapping có thể phân kỳ. DQN giảm nhẹ bằng mạng đích + replay; đừng bỏ cái nào.
- **Khám phá.** ε phải giảm dần, thường từ 1.0 xuống 0.01 trong ~10% đầu quá trình huấn luyện. Không đủ khám phá sớm thì Q-net sẽ hội tụ vào một bồn cục bộ.
- **Ước lượng quá cao.** Lấy `max` trên Q nhiễu bị chệch lên trên. Luôn dùng Double DQN khi chạy thật.
- **Thang phần thưởng.** Cắt hoặc chuẩn hoá phần thưởng; độ lớn gradient tỉ lệ với độ lớn phần thưởng.
- **Khởi động nguội bộ đệm phát lại.** Đừng huấn luyện cho tới khi bộ đệm có vài nghìn bước chuyển. Gradient sớm trên ~20 mẫu sẽ overfit.
- **Tần suất đồng bộ mạng đích.** Quá thường xuyên ≈ không có mạng đích; quá thưa ≈ mục tiêu cũ. Atari DQN dùng 10,000 bước môi trường. Quy tắc ngón tay cái: đồng bộ mỗi ~1/100 chân trời huấn luyện.
- **Tiền xử lý quan sát.** Atari DQN chồng 4 khung hình để state có tính Markov. Mọi môi trường có thông tin vận tốc đều cần chồng khung hình hoặc state hồi quy.

## Áp dụng

Năm 2026, DQN hiếm khi còn là state-of-the-art nhưng vẫn là thuật toán off-policy tham chiếu:

| Bài toán | Phương pháp nên chọn | Vì sao không dùng DQN? |
|------|------------------|--------------|
| Giống Atari với hành động rời rạc | Rainbow DQN hoặc Muesli | Cùng khung, nhiều mẹo hơn. |
| Điều khiển liên tục | SAC / TD3 (Giai đoạn 9 · 07) | DQN không có mạng chính sách. |
| On-policy / thông lượng cao | PPO (Giai đoạn 9 · 08) | Không có replay buffer; dễ mở rộng hơn. |
| RL offline | CQL / IQL / Decision Transformer | Mục tiêu Q thận trọng, không bị nổ do bootstrapping. |
| Không gian hành động rời rạc lớn (gợi ý) | DQN với action embedding, hoặc IMPALA | Ổn; phần trang trí mới là quan trọng. |
| RL cho LLM | PPO / GRPO | Ở mức chuỗi, không phải mức bước; loss khác. |

Các bài học vẫn còn giá trị. Replay và mạng đích xuất hiện trong SAC, TD3, DDPG, SAC-X, bộ đệm self-play của AlphaZero, và mọi phương pháp RL offline. Reward clipping tiếp tục sống dưới dạng chuẩn hoá advantage trong PPO. Kiến trúc này là bản thiết kế mẫu.

## Đóng gói

Lưu thành `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Vẽ đường cong return theo từng episode. Mất bao nhiêu episode để trung bình trượt vượt -10?
2. **Trung bình.** Tắt mạng đích (dùng mạng online cho cả hai vế của mục tiêu Bellman). Đo độ bất ổn khi huấn luyện — return có dao động hay phân kỳ không?
3. **Khó.** Thêm Double DQN: dùng mạng online để chọn `argmax a'`, mạng đích để đánh giá. So sánh độ chệch của `Q(s_0, best_a)` với `V*(s_0)` thật sau 1,000 episode, có và không có Double DQN, trên một GridWorld có phần thưởng nhiễu.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning với hàm Q dạng nơ-ron, bộ đệm phát lại và mạng đích. |
| Experience replay | "Các bước chuyển bị xáo trộn" | Ring buffer được lấy mẫu đều ở mỗi bước gradient; làm mất tương quan dữ liệu. |
| Target network | "Bootstrap bị đóng băng" | Bản sao định kỳ của Q dùng trong mục tiêu Bellman; giúp huấn luyện ổn định. |
| Deadly triad | "Vì sao RL phân kỳ" | Xấp xỉ hàm + bootstrapping + off-policy = không có bảo đảm hội tụ. |
| Double DQN | "Cách sửa độ chệch tối đa hoá" | Mạng online chọn hành động, mạng đích đánh giá nó. |
| Dueling DQN | "Đầu V và A" | Phân rã Q = V + A - mean(A); cùng đầu ra, gradient chảy tốt hơn. |
| Rainbow | "Tất cả các mẹo" | DDQN + PER + dueling + n-step + noisy + distributional trong một. |
| PER | "Prioritized Replay" | Lấy mẫu bước chuyển tỉ lệ với độ lớn TD-error. |

## Đọc thêm

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — bài báo hội thảo NeurIPS 2013 khởi động deep RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — bài báo trên Nature, DQN 49 trò chơi.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — bài báo chồng các mẹo.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — phần sách giáo khoa về "bộ ba chết người" (xấp xỉ hàm + bootstrapping + off-policy) mà mạng đích và bộ đệm phát lại của DQN được thiết kế để chế ngự.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — bản DQN một file tham chiếu dùng trong các nghiên cứu ablation; đáng đọc song song với bản tự viết của bài này.

---

[← Bài trước: Temporal Difference](../../04-q-learning-sarsa/docs/vi.md) · [Bài sau: Policy Gradient →](../../06-policy-gradients-reinforce/docs/vi.md)
