# RL cho game — AlphaZero, MuZero và kỷ nguyên suy luận của LLM

> 1992: TD-Gammon đánh bại các nhà vô địch backgammon bằng TD thuần. 2016: AlphaGo đánh bại Lee Sedol. 2017: AlphaZero thống trị cờ vua, shogi và Go từ con số không. 2024: DeepSeek-R1 chứng minh công thức tương tự, với GRPO thay cho PPO, cũng hiệu quả với suy luận. Game là thước đo thúc đẩy mọi bước đột phá trong giai đoạn này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 05 (DQN), Giai đoạn 9, Bài 08 (PPO), Giai đoạn 9, Bài 09 (RLHF), Giai đoạn 9, Bài 10 (MARL)
**Time:** ~120 phút

## Vấn đề

Game playing (chơi game) có mọi thứ mà RL cần. Phần thưởng sạch (thắng/thua). Episode vô hạn (self-play (tự đấu) reset lại). Mô phỏng hoàn hảo (trò chơi *chính là* trình mô phỏng). Không gian hành động rời rạc hoặc liên tục nhỏ. Cấu trúc đa tác tử buộc mô hình phải chịu được đối kháng.

Và game là nơi mọi đột phá lớn của RL được thử nghiệm. TD-Gammon (backgammon, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Five (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (mô hình học được, 2019). AlphaTensor (nhân ma trận, 2022). AlphaDev (thuật toán sắp xếp, 2023). DeepSeek-R1 (suy luận toán, 2025) — minh chứng mới nhất rằng kỹ thuật RL cho game cũng hiệu quả trên văn bản.

Bài capstone này khảo sát ba kiến trúc mốc — AlphaZero, MuZero và GRPO — qua một lăng kính thống nhất: **self-play + search (tìm kiếm) + cải thiện policy**. Mỗi kiến trúc tổng quát hoá cái trước; GRPO đặc biệt là công thức của AlphaZero áp dụng cho suy luận của LLM, với token là hành động và kiểm chứng toán học là tín hiệu thắng.

## Khái niệm

![AlphaZero ↔ MuZero ↔ GRPO: cùng một vòng lặp, môi trường khác nhau](../assets/rl-games.svg)

**Vòng lặp thống nhất.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver và cộng sự. Cho một game (cờ vua, shogi, Go) đã biết luật:

- Policy-value network (mạng chính sách và mạng giá trị): một tower `f_θ(s) → (p, v)`. `p` là prior trên các nước đi hợp lệ. `v` là kết quả kỳ vọng của ván đấu.
- Monte Carlo Tree Search (MCTS, tìm kiếm trên cây Monte Carlo): mỗi nước đi, mở một cây các khả năng tiếp diễn. Dùng `(p, v)` làm prior + bootstrapping (lan giá trị từ các ước lượng khác). Chọn nút theo UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Self-play: cho agent đấu với chính nó. Ở nước `t`, phân phối số lần thăm của MCTS `π_t` trở thành mục tiêu huấn luyện của policy.
- Loss: `L = (v - z)² - π · log p + c · ||θ||²`. `z` là kết quả ván đấu (+1 / 0 / -1).

Không kiến thức con người. Không heuristic thủ công, không định hình phần thưởng (reward shaping). Một công thức duy nhất chinh phục cờ vua, shogi và Go sau vài chục triệu ván self-play mỗi trò.

**MuZero (2019).** Schrittwieser và cộng sự. Bỏ yêu cầu phải biết luật.

- Thay vì môi trường cố định, học một *mô hình động lực ẩn* `(h, g, f)`:
  - `h(s)`: mã hoá quan sát thành trạng thái ẩn.
  - `g(s_latent, a)`: dự đoán trạng thái ẩn tiếp theo + phần thưởng.
  - `f(s_latent)`: dự đoán prior của policy + giá trị.
- MCTS chạy trong *không gian ẩn đã học*. Cùng cách tìm kiếm, cùng vòng huấn luyện.
- Chạy được trên Go, cờ vua, shogi *và* Atari — một thuật toán, không cần biết luật.

**Stochastic MuZero (2022).** Thêm động lực ngẫu nhiên và nút ngẫu nhiên; mở rộng sang các game kiểu backgammon.

**Muesli, Gumbel MuZero (2022-2024).** Cải thiện hiệu quả mẫu và tìm kiếm tất định.

**GRPO (2024-2025).** Công thức của DeepSeek-R1. Cùng vòng lặp hình AlphaZero, áp dụng cho suy luận của mô hình ngôn ngữ:

- "Game": trả lời một bài toán / lập trình / suy luận. "Thắng" = verifier trả về 1 (test case pass, đáp án số khớp).
- Policy: LLM. Hành động: token. Trạng thái: prompt + phần trả lời tính tới thời điểm hiện tại.
- Không có critic (V_φ kiểu PPO). Thay vào đó, với mỗi prompt, lấy mẫu `G` completion từ policy. Tính phần thưởng cho từng cái. Dùng **group-relative advantage** (lợi thế tương đối theo nhóm) `A_i = (r_i - mean_r) / std_r` làm tín hiệu cho cập nhật kiểu REINFORCE.
- Phạt KL so với policy tham chiếu để chống trôi (như RLHF).
- Loss đầy đủ:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Không reward model, không critic, không MCTS. Baseline tương đối theo nhóm thay thế cả ba. Ngang hoặc vượt chất lượng PPO-RLHF trên các benchmark suy luận với một phần nhỏ compute.

**Công thức R1 đầy đủ.** DeepSeek-R1 (DeepSeek 2025) là hai mô hình trong một bài báo:

- **R1-Zero.** Bắt đầu từ mô hình nền DeepSeek-V3. Không SFT. Áp GRPO trực tiếp với hai thành phần phần thưởng: *accuracy reward* (dựa trên luật — đáp án cuối có parse ra đúng số không / code có pass unit test không) và *format reward* (completion có bọc chuỗi suy luận trong thẻ `<think>…</think>` không). Qua hàng nghìn bước, độ dài trả lời trung bình tăng từ ~100 lên ~10,000 token và điểm benchmark toán leo tới mức gần o1-preview. Mô hình tự học suy luận từ con số không. Nhược điểm: chuỗi suy luận của nó thường khó đọc, trộn ngôn ngữ và thiếu trau chuốt văn phong.
- **R1.** Khắc phục vấn đề dễ đọc của R1-Zero bằng pipeline bốn giai đoạn:
  1. **Cold-start SFT.** Thu vài nghìn minh hoạ CoT dài với định dạng sạch. Supervised-finetune mô hình nền trên đó. Đây là điểm khởi đầu dễ đọc.
  2. **GRPO hướng suy luận.** Áp GRPO với phần thưởng accuracy+format cùng một phần thưởng *nhất quán ngôn ngữ* để chặn code-switching.
  3. **Rejection sampling + SFT vòng 2.** Lấy mẫu ~600K quỹ đạo suy luận từ checkpoint RL, chỉ giữ những cái có đáp án cuối đúng và CoT dễ đọc, rồi kết hợp với ~200K ví dụ SFT không-suy-luận (viết, QA, self-cognition). Fine-tune lại mô hình nền.
  4. **GRPO toàn phổ.** Thêm một vòng RL nữa, phủ cả suy luận (phần thưởng dựa trên luật) và căn chỉnh tổng quát (phần thưởng dựa trên sở thích: hữu ích/vô hại).

Kết quả ngang o1 trên AIME và MATH-500 với trọng số mở, và đủ nhỏ để distill. Cùng bài báo cũng phát hành sáu mô hình dense đã distill (Qwen-1.5B tới Llama-70B) bằng cách SFT trên các vết suy luận của R1 — không RL ở mô hình học trò. Distill từ một teacher RL mạnh luôn thắng RL from scratch ở quy mô của học trò.

**Vì sao dùng GRPO thay PPO cho suy luận.** Ba lý do trong bài báo DeepSeekMath (2/2024): (1) không phải huấn luyện value network, tiết kiệm một nửa bộ nhớ; (2) baseline theo nhóm xử lý tự nhiên phần thưởng thưa ở cuối quỹ đạo mà các tác vụ suy luận tạo ra; (3) chuẩn hoá theo từng prompt khiến lợi thế so sánh được giữa các bài toán khó dễ rất khác nhau, điều mà một critic duy nhất của PPO không làm được.

**Không tìm kiếm vs có tìm kiếm.** Game đã phân nhánh:

- *Game thông tin hoàn hảo với chân trời dài* (Go, cờ vua): vẫn dựa trên tìm kiếm. AlphaZero / MuZero thống trị.
- *Suy luận LLM*: chưa có MCTS trong production; GRPO trên full rollout, best-of-N cho compute lúc inference. Process reward model (PRM) gợi ý rằng tìm kiếm theo bước đang được đưa trở lại.

```figure
f3-selfplay-ladder
```

## Tự tay làm

Code trong `code/main.py` triển khai **GRPO thu nhỏ** — một bandit với nhiều nhóm mẫu. Thuật toán giống hệt như trên LLM; chỉ policy và môi trường là đơn giản hơn. Nó dạy *loss* và *group-relative advantage*, thứ đổi mới của năm 2025.

### Bước 1: một môi trường verifier tí hon

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

Trong GRPO thật, verifier chạy unit test hoặc kiểm tra đẳng thức toán học.

### Bước 2: policy: softmax trên K token đáp án cho mỗi prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Tương đương output của lớp cuối trong LLM khi có điều kiện là một prompt.

### Bước 3: lấy mẫu theo nhóm và lợi thế tương đối theo nhóm

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage là mẹo DeepSeek 2024. Không cần critic. "Baseline" là trung bình nhóm, và chuẩn hoá dùng độ lệch chuẩn của nhóm.

### Bước 4: so sánh với baseline REINFORCE (không value)

Cùng thiết lập, cùng compute, REINFORCE thuần. GRPO hội tụ nhanh hơn và ổn định hơn.

### Bước 5: quan sát entropy và KL

Cùng bộ chẩn đoán như RLHF: KL trung bình tới policy tham chiếu, entropy của policy, phần thưởng theo thời gian. Khi các chỉ số này ổn định, huấn luyện xong.

## Cạm bẫy

- **Reward hacking nhờ lách verifier.** GRPO thừa hưởng rủi ro của RLHF: nếu verifier sai hoặc bị lách được, LLM sẽ tìm ra chỗ lách. Verifier vững (nhiều test case, chứng minh hình thức) rất quan trọng.
- **Nhóm quá nhỏ.** Phương sai của baseline theo nhóm xấp xỉ `1/√G`. Dưới `G = 4`, tín hiệu lợi thế nhiễu; lựa chọn chuẩn là `G = 8` tới `64`.
- **Thiên lệch độ dài.** Completion của LLM có độ dài khác nhau thì log-probability khác nhau. Chuẩn hoá theo số token, hoặc dùng log-prob ở mức chuỗi, hoặc cắt về độ dài tối đa.
- **Vòng self-play thuần.** Huấn luyện kiểu AlphaZero có thể kẹt trong vòng lặp thống trị ở các game tổng-khác-không. Giảm nhẹ bằng bể đối thủ đa dạng (league play, Bài 10).
- **Lệch giữa tìm kiếm và policy.** AlphaZero huấn luyện policy bắt chước output của tìm kiếm. Nếu mạng policy quá nhỏ để biểu diễn phân phối của tìm kiếm, huấn luyện đình trệ.
- **Sàn compute.** MuZero / AlphaZero cần compute khổng lồ. Một ablation thường tốn hàng trăm GPU-giờ. Có các demo thu nhỏ (ví dụ AlphaZero trên Connect Four) để học.
- **Độ phủ của verifier.** Unit test pass cho một lời giải lỗi sẽ củng cố cái lỗi đó. Thiết kế verifier bắt được các ca biên.

## Áp dụng

Bức tranh RL cho game năm 2026, theo lĩnh vực:

| Lĩnh vực | Phương pháp chiếm ưu thế |
|--------|-----------------|
| Game bàn hai người tổng-bằng-không (Go, cờ vua, shogi) | AlphaZero / MuZero / KataGo |
| Game bài thông tin không hoàn hảo (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / game pixel | Muesli / MuZero / IMPALA-PPO |
| Chiến lược nhiều người chơi quy mô lớn (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| Suy luận toán/code của LLM | GRPO (DeepSeek-R1, Qwen-RL, các bản tái lập mở) |
| Căn chỉnh LLM | DPO / RLHF-PPO (không phải GRPO; verifier là sở thích, không kiểm chứng được) |
| Robotics | PPO + DR (không phải RL cho game, nhưng dùng chung công cụ policy-gradient) |
| Bài toán tổ hợp | Các biến thể AlphaZero (AlphaTensor, AlphaDev) |

Cái *công thức* — self-play, cải thiện có tìm kiếm hỗ trợ, distill policy — trải khắp văn bản, pixel và điều khiển vật lý. GRPO là hiện thân trẻ nhất; sẽ còn nhiều hơn nữa.

## Đóng gói

Lưu thành `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Bài tập

1. **Dễ.** Cài đặt bandit GRPO trong `code/main.py`. Huấn luyện trên 2 prompt × 4 token đáp án mỗi prompt. Hội tụ trong < 1,000 cập nhật với `G=8`.
2. **Trung bình.** Cắm PPO (clipped) và REINFORCE gốc vào. So sánh hiệu quả mẫu và phương sai phần thưởng với GRPO trên cùng bandit.
3. **Khó.** Mở rộng thành "chuỗi suy luận" độ dài 2: agent phát hai token và verifier thưởng cho cả cặp. Đo cách GRPO xử lý gán công trạng qua chuỗi hai bước. (Gợi ý: tính lợi thế nhóm cho *toàn chuỗi*, lan sang cả hai vị trí token.)

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| MCTS | "Tìm kiếm trên cây với mạng đã học" | Monte Carlo Tree Search; chọn theo UCB1/PUCT với prior `(p, v)` đã học. |
| AlphaZero | "Self-play + MCTS" | Mạng policy-value huấn luyện để khớp số lần thăm của MCTS và kết quả ván đấu. |
| MuZero | "AlphaZero với mô hình học được" | Cùng vòng lặp nhưng trong không gian ẩn nhờ động lực học được. |
| GRPO | "PPO không critic" | Group Relative Policy Optimization; REINFORCE với baseline là trung bình nhóm + KL. |
| PUCT | "UCB của AlphaZero" | `Q + c · p · √N / (1 + N_a)` — cân bằng ước lượng giá trị với prior. |
| Self-play | "Agent đấu với chính nó trong quá khứ" | Chuẩn cho tổng-bằng-không; tín hiệu huấn luyện đối xứng. |
| League play | "Self-play dựa trên quần thể" | Lấy mẫu đối thủ từ các phiên bản cũ + hiện tại + exploiter. |
| Verifier reward | "RL kiểm chứng được" | Phần thưởng đến từ bộ kiểm tra tất định (test pass, đáp án khớp). |
| Process reward | "PRM" | Chấm điểm từng bước suy luận, không chỉ đáp án cuối cùng. |

## Đọc thêm

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — bài báo giới thiệu GRPO và baseline tương đối theo nhóm.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — công thức R1 bốn giai đoạn đầy đủ cùng ablation R1-Zero.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep learning ở quy mô lớn.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — bài báo khởi đầu tất cả.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — tài liệu tham chiếu production để áp GRPO với hàm phần thưởng tuỳ biến.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — bản tái lập mở công thức R1 ở nhiều quy mô.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — cách đặt vấn đề từ sách giáo trình cho self-play, tìm kiếm và "designed reward" mà R1 hiện thực ở quy mô LLM.

---

[← Bài trước: Chuyển giao (transfer) sim-to-real (từ mô phỏng ra thực tế)](../../11-sim-to-real-transfer/docs/vi.md)
