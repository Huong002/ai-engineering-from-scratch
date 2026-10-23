# RL đa tác tử

> RL một tác tử giả định môi trường là dừng. Đặt hai tác tử đang học vào cùng một thế giới thì giả định đó sụp đổ: mỗi tác tử là một phần môi trường của tác tử kia, và cả hai đều đang thay đổi. Multi-agent RL (RL đa tác tử) là tập hợp các thủ thuật để việc học hội tụ khi giả định Markov không còn đúng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 04 (Q-learning), Giai đoạn 9, Bài 06 (REINFORCE), Giai đoạn 9, Bài 07 (Actor-Critic)
**Time:** ~45 phút

## Vấn đề

Một con robot học cách đi trong một căn phòng là bài toán RL một tác tử. Một đội bóng đá thì không. AlphaStar đấu với đối thủ StarCraft thì không. Một chợ gồm các tác tử đặt giá thì không. Hai chiếc xe nhường nhau ở ngã tư bốn hướng thì không. Rất nhiều bài toán thực tế kiểu nhiều-đấu-nhiều thì không.

Trong mọi bối cảnh đa tác tử, từ góc nhìn của bất kỳ một tác tử nào, các tác tử khác *chính là* một phần của môi trường. Khi chúng học và thay đổi hành vi, môi trường trở nên không dừng. Tính chất Markov — "trạng thái kế tiếp chỉ phụ thuộc trạng thái hiện tại và hành động của tôi" — bị vi phạm, vì trạng thái kế tiếp còn phụ thuộc vào lựa chọn của các tác tử *khác*, và chính sách của chúng là những mục tiêu đang di chuyển.

Điều này phá vỡ các chứng minh hội tụ dạng bảng (đảm bảo của Q-learning giả định môi trường dừng). Nó cũng phá vỡ deep RL ngây thơ: các tác tử rượt đuổi nhau theo vòng lặp, không bao giờ hội tụ về một chính sách ổn định. Bạn cần các kỹ thuật dành riêng cho đa tác tử: huấn luyện tập trung / thực thi phi tập trung, baseline phản thực tế, league play, self-play (tự đấu).

Ứng dụng năm 2026: bầy robot, điều phối giao thông, đội xe tự hành, mô phỏng thị trường, hệ thống LLM đa tác tử (Giai đoạn 16), và bất kỳ trò chơi nào có nhiều hơn một người chơi thông minh.

## Khái niệm

![Bốn chế độ MARL: độc lập, critic tập trung, self-play, league](../assets/marl.svg)

**Hình thức hoá: Markov game.** Một dạng tổng quát của MDP: trạng thái `S`, hành động phối hợp `a = (a_1, …, a_n)`, chuyển trạng thái `P(s' | s, a)`, và phần thưởng riêng cho từng tác tử `R_i(s, a, s')`. Mỗi tác tử `i` tối đa hoá lợi tức của mình theo chính sách `π_i` của nó. Nếu phần thưởng giống hệt nhau, nó là **fully cooperative** (hợp tác hoàn toàn). Nếu zero-sum, nó là **adversarial** (đối kháng). Nếu hỗn hợp, nó là **general-sum** (tổng khác không).

**Các thách thức cốt lõi:**

- **Non-stationarity (môi trường không dừng).** `P(s' | s, a_i)` từ góc nhìn của tác tử `i` phụ thuộc vào `π_{-i}`, mà cái đó đang thay đổi.
- **Credit assignment (quy công).** Với phần thưởng chia sẻ, tác tử nào đã tạo ra nó?
- **Phối hợp khám phá.** Các tác tử phải khám phá những chiến lược bổ trợ nhau, chứ không khám phá trùng lặp cùng một trạng thái.
- **Khả năng mở rộng.** Không gian hành động phối hợp tăng theo hàm mũ theo `n`.
- **Quan sát một phần.** Mỗi tác tử chỉ thấy quan sát của riêng nó; trạng thái toàn cục bị ẩn.

**Bốn chế độ chủ đạo:**

**1. Q-learning độc lập / PPO độc lập (IQL, IPPO).** Mỗi tác tử học Q hoặc chính sách của riêng nó, coi những tác tử khác là một phần môi trường. Đơn giản, đôi khi hiệu quả (nhất là khi experience replay đóng vai trò một thủ thuật làm mượt, mô hình hoá đối thủ). Hội tụ lý thuyết: không có. Thực tế: ổn với các nhiệm vụ liên kết lỏng, tệ với các nhiệm vụ liên kết chặt.

**2. Huấn luyện tập trung, thực thi phi tập trung (CTDE).** Mô hình phổ biến nhất hiện nay. Mỗi tác tử có *chính sách* `π_i` riêng, điều kiện hoá trên quan sát cục bộ `o_i` — thực thi phi tập trung chuẩn khi triển khai. Trong lúc *huấn luyện*, một critic tập trung `Q(s, a_1, …, a_n)` điều kiện hoá trên toàn bộ trạng thái toàn cục và hành động phối hợp. Ví dụ:
- **MADDPG** (Lowe et al. 2017): DDPG với một critic tập trung cho mỗi tác tử.
- **COMA** (Foerster et al. 2017): baseline phản thực tế — hỏi "phần thưởng của tôi sẽ là bao nhiêu nếu tôi chọn hành động `a'` thay vào đó?" — để tách riêng đóng góp của tôi.
- **MAPPO** / **IPPO** với critic chia sẻ (Yu et al. 2022): PPO với một hàm giá trị tập trung. Thống trị năm 2026 với MARL hợp tác.
- **QMIX** (Rashid et al. 2018): phân rã giá trị — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` với mixing đơn điệu.

**3. Self-play.** Hai bản sao của cùng một tác tử chơi với nhau. Chính sách của đối thủ *chính là* chính sách của tôi ở một bản chụp trong quá khứ. AlphaGo / AlphaZero / MuZero. OpenAI Five. Hiệu quả nhất với trò chơi zero-sum; tín hiệu huấn luyện là đối xứng.

**4. League play.** Một phần mở rộng của self-play sang môi trường tổng hỗn hợp / đối kháng: giữ một quần thể các chính sách quá khứ và hiện tại, lấy mẫu một đối thủ từ league, rồi huấn luyện đấu với chúng. Thêm các exploiter (chuyên đánh bại kẻ mạnh nhất hiện tại) và main exploiter (chuyên đánh bại các exploiter). AlphaStar (StarCraft II). Cần thiết khi trò chơi có những chu trình chiến lược kiểu "oẳn tù tì".

**Giao tiếp.** Cho phép các tác tử gửi những thông điệp đã học `m_i` cho nhau. Hiệu quả trong bối cảnh hợp tác. Foerster et al. (2016) chỉ ra rằng giao tiếp giữa các tác tử dạng khả vi có thể được huấn luyện đầu-cuối. Các hệ thống đa tác tử dựa trên LLM ngày nay (Giai đoạn 16) về bản chất giao tiếp bằng ngôn ngữ tự nhiên.

```figure
f3-marl-orbit
```

## Tự tay làm

Bài này dùng một GridWorld 6×6 với hai tác tử hợp tác. Chúng xuất phát ở hai góc đối diện và phải tới một đích chung. Phần thưởng chia sẻ: `-1` mỗi bước chừng nào còn một tác tử đang di chuyển, `+10` khi cả hai đều tới. Xem `code/main.py`.

### Bước 1: môi trường đa tác tử

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

Không gian hành động *phối hợp* là `|A|² = 16`. Trạng thái toàn cục là hai vị trí.

### Bước 2: Q-learning độc lập

Mỗi tác tử chạy Q-table riêng của mình, khoá theo trạng thái phối hợp. Ở mỗi bước: cả hai chọn hành động ε-greedy, thu thập chuyển trạng thái phối hợp, mỗi tác tử cập nhật Q của mình bằng phần thưởng chia sẻ.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Chạy được trên nhiệm vụ này vì phần thưởng dày và đồng hướng. Thất bại trên các nhiệm vụ liên kết chặt (ví dụ: một tác tử phải *chờ* tác tử kia).

### Bước 3: Q tập trung với cập nhật phân rã giá trị

Dùng một Q trên hành động phối hợp `Q(s, a_1, a_2)`. Cập nhật từ phần thưởng chia sẻ. Phi tập trung hoá khi thực thi bằng cách lấy biên: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Đổi không gian hành động phối hợp cấp số mũ lấy một cái nhìn toàn cục *đúng*.

### Bước 4: self-play đơn giản (2 tác tử đối kháng)

Cùng một tác tử, hai vai. Huấn luyện tác tử A đấu với tác tử B; sau `K` episode, sao chép trọng số của A sang B. Huấn luyện đối xứng, tiến bộ nhất quán. Công thức AlphaZero thu nhỏ.

## Cạm bẫy

- **Replay không dừng.** Experience replay với các tác tử độc lập còn tệ hơn với một tác tử, vì các chuyển trạng thái cũ được sinh ra bởi những đối thủ nay đã lỗi thời. Cách sửa: gán lại nhãn hoặc đánh trọng số theo độ mới.
- **Quy công mơ hồ.** Phần thưởng chia sẻ sau một episode dài; không có cách rõ ràng để nói tác tử nào đã đóng góp. Cách sửa: baseline phản thực tế (COMA), hoặc định hình phần thưởng theo từng tác tử.
- **Chính sách trôi / rượt đuổi.** Phản hồi tốt nhất của mỗi tác tử thay đổi theo từng cập nhật của tác tử kia. Cách sửa: critic tập trung, tốc độ học chậm, hoặc đóng băng từng tác tử một.
- **Reward hacking nhờ phối hợp.** Các tác tử tìm ra những lỗ hổng phối hợp mà người thiết kế không lường trước. Các tác tử đấu giá hội tụ về đặt giá bằng không. Cách sửa: thiết kế phần thưởng cẩn thận, ràng buộc hành vi.
- **Khám phá trùng lặp.** Cả hai tác tử khám phá cùng những cặp trạng thái-hành động. Cách sửa: thưởng entropy cho từng tác tử, hoặc điều kiện hoá theo vai.
- **Chu trình league.** Self-play thuần có thể kẹt trong một chu trình thống trị. Cách sửa: league play với đối thủ đa dạng.
- **Bùng nổ mẫu.** `n` tác tử × không gian trạng thái × hành động phối hợp. Xấp xỉ bằng hàm xấp xỉ; không gian hành động dạng phân rã (một đầu ra chính sách cho mỗi tác tử).

## Áp dụng

Bản đồ ứng dụng MARL năm 2026:

| Lĩnh vực | Phương pháp | Ghi chú |
|--------|--------|-------|
| Điều hướng / thao tác hợp tác | MAPPO / QMIX | CTDE; critic chia sẻ + actor phi tập trung. |
| Trò chơi hai người (cờ vua, Go, poker) | Self-play với MCTS (AlphaZero) | Zero-sum; huấn luyện đối xứng. |
| Nhiều người chơi phức tạp (Dota, StarCraft) | League play + tiền huấn luyện bằng bắt chước | OpenAI Five, AlphaStar. |
| Đội xe tự hành | CTDE MAPPO / PPO với attention | Quan sát một phần; kích thước đội thay đổi. |
| Thị trường đấu giá | Cân bằng lý thuyết trò chơi + RL | Mean-field RL khi `n` → ∞. |
| Hệ thống LLM đa tác tử (Giai đoạn 16) | Giao tiếp bằng ngôn ngữ tự nhiên + điều kiện hoá theo vai | Vòng RL ở tầng lập kế hoạch của tác tử. |

Năm 2026, mảng tăng trưởng lớn nhất của MARL là dựa trên LLM: bầy tác tử mô hình ngôn ngữ đàm phán, tranh luận, xây dựng phần mềm. RL xuất hiện dưới dạng tối ưu hoá sở thích trên đầu ra *cấp độ quỹ đạo*, không phải cấp độ token (Giai đoạn 16 · 03).

## Đóng gói

Lưu thành `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Bài tập

1. **Dễ.** Huấn luyện Q-learning độc lập trên GridWorld hợp tác 2 tác tử. Mất bao nhiêu episode cho tới khi lợi tức trung bình > 0? Vẽ đường cong học phối hợp.
2. **Trung bình.** Thêm một nhiệm vụ "phối hợp": chỉ tới đích khi cả hai tác tử bước lên đích trong cùng một lượt. Q độc lập còn hội tụ không? Cái gì vỡ?
3. **Khó.** Cài đặt một critic tập trung cho kiểu huấn luyện MAPPO và so sánh tốc độ hội tụ với PPO độc lập trên nhiệm vụ phối hợp.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Markov game | "MDP đa tác tử" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; mỗi tác tử có phần thưởng riêng. |
| CTDE | "Huấn luyện tập trung, thực thi phi tập trung" | Critic phối hợp lúc huấn luyện; chính sách mỗi tác tử chỉ dùng quan sát cục bộ. |
| IPPO | "PPO độc lập" | Mỗi tác tử chạy PPO riêng. Baseline đơn giản; thường bị đánh giá thấp. |
| MAPPO | "PPO đa tác tử" | PPO với một hàm giá trị tập trung, điều kiện hoá trên trạng thái toàn cục. |
| QMIX | "Phân rã giá trị đơn điệu" | `Q_tot = f_monotone(Q_1, …, Q_n)` cho phép argmax phi tập trung. |
| COMA | "Đa tác tử phản thực tế" | Advantage = Q của tôi trừ Q kỳ vọng khi lấy biên theo hành động của tôi. |
| Self-play | "Tác tử đấu với chính nó trong quá khứ" | Một tác tử, hai vai; chuẩn cho trò chơi zero-sum. |
| League play | "Huấn luyện theo quần thể" | Lưu chính sách quá khứ, lấy mẫu đối thủ từ pool; xử lý chu trình chiến lược. |

## Đọc thêm

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE với một critic tập trung.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — baseline phản thực tế cho quy công.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — phân rã giá trị với tính đơn điệu.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO mạnh bất ngờ cho MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — league play ở quy mô lớn.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — self-play thuần trong trò chơi zero-sum.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — bao gồm phần trình bày ngắn của sách giáo khoa về bối cảnh đa tác tử và vấn đề môi trường không dừng mà CTDE được thiết kế để giải quyết.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — bài khảo sát bao quát MARL hợp tác, cạnh tranh và hỗn hợp kèm các kết quả hội tụ.

---

[← Bài trước: Mô hình hóa phần thưởng & RLHF](../../09-reward-modeling-rlhf/docs/vi.md) · [Bài sau: Chuyển giao (transfer) sim-to-real (từ mô phỏng ra thực tế) →](../../11-sim-to-real-transfer/docs/vi.md)
