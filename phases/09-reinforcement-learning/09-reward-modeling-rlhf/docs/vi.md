# Mô hình hóa phần thưởng & RLHF

> Con người không thể viết một hàm phần thưởng cho "câu trả lời trợ lý tốt", nhưng họ có thể so sánh hai câu trả lời và chọn câu tốt hơn. Hãy khớp một mô hình phần thưởng với những so sánh đó, rồi dùng RL để huấn luyện mô hình ngôn ngữ theo nó. Christiano 2017. InstructGPT 2022. Công thức đã biến GPT-3 thành ChatGPT. Năm 2026 nó phần lớn đang được thay bằng DPO — nhưng mô hình tư duy thì vẫn còn nguyên.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 05 (Phân tích cảm xúc), Giai đoạn 9, Bài 08 (PPO)
**Time:** ~45 phút

## Vấn đề

Bạn đã huấn luyện một mô hình ngôn ngữ theo mục tiêu dự đoán token tiếp theo. Nó viết tiếng Anh đúng ngữ pháp. Nó cũng nói dối, lan man và từ chối cả việc từ chối. Bạn không thể sửa điều này bằng cách pretrain thêm — văn bản web chính là vấn đề, không phải liều thuốc.

Bạn muốn một *phần thưởng vô hướng* nói rằng "câu trả lời A tốt hơn câu trả lời B cho chỉ dẫn X". Viết hàm phần thưởng đó bằng tay là bất khả thi. "Tính hữu ích" không phải một biểu thức dạng đóng trên các token. Nhưng con người có thể so sánh hai đầu ra và đánh dấu một sở thích (preference). Thu thập ở quy mô lớn thì rẻ.

RLHF (Reinforcement Learning from Human Feedback — học tăng cường từ phản hồi từ con người; Christiano et al. 2017; Ouyang et al. 2022) biến sở thích thành một reward model (mô hình phần thưởng), rồi tối ưu LM bằng PPO theo phần thưởng đó. Ba bước: SFT → RM → PPO. Đây là công thức đã đưa ChatGPT, Claude, Gemini và mọi aligned-LLM khác ra đời trong 2023–2025.

Năm 2026, bước PPO phần lớn được thay bằng DPO (Giai đoạn 10 · 08) vì rẻ hơn và gần như tốt ngang cho việc tinh chỉnh alignment. Nhưng phần *mô hình phần thưởng* vẫn là nền của mọi bộ lấy mẫu Best-of-N, mọi pipeline RL-from-verifiable-rewards, và mọi mô hình suy luận dùng process reward model. Hiểu RLHF là hiểu toàn bộ chồng công nghệ alignment.

## Khái niệm

![Ba giai đoạn RLHF: SFT, huấn luyện RM trên sở thích theo cặp, PPO với hình phạt KL](../assets/rlhf.svg)

**Giai đoạn 1: Supervised Fine-Tuning (SFT).** Bắt đầu từ một mô hình nền đã pretrain. Fine-tune trên các minh họa do con người viết về hành vi mục tiêu (câu trả lời tuân theo chỉ dẫn, phản hồi hữu ích, v.v.). Kết quả: một mô hình `π_SFT` *thiên về hành vi tốt* nhưng vẫn có không gian hành động không bị chặn.

**Giai đoạn 2: Huấn luyện mô hình phần thưởng.**

- Thu thập các cặp câu trả lời `(y_+, y_-)` cho các prompt `x`, được con người gán nhãn "y_+ được ưa thích hơn y_-".
- Huấn luyện một mô hình phần thưởng `R_φ(x, y)` để gán điểm cao hơn cho `y_+`.
- Loss: **logistic Bradley-Terry cho so sánh theo cặp (pairwise comparison)**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ là hàm sigmoid. Chênh lệch phần thưởng ngụ ý log-odds của sở thích. BT — Bradley-Terry (mô hình so sánh theo cặp) — đã là chuẩn từ 1952 và là lựa chọn thống trị trong RLHF hiện đại.

- `R_φ` thường được khởi tạo từ mô hình SFT với một head vô hướng ở trên. Cùng backbone transformer; một lớp tuyến tính duy nhất xuất ra phần thưởng.

**Giai đoạn 3: PPO chống lại RM với hình phạt KL.**

- Khởi tạo policy huấn luyện được `π_θ` từ `π_SFT`. Giữ một *reference* đóng băng `π_ref = π_SFT`.
- Phần thưởng ở cuối câu trả lời `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  Hình phạt KL divergence (phân kỳ KL) ngăn `π_θ` trôi tùy tiện khỏi `π_SFT` — nó là một *bộ điều chuẩn*, không phải một trust region cứng. `β` thường là `0.01`-`0.05`.
- Chạy PPO (Bài 08) với phần thưởng này. Advantage được tính trên quỹ đạo mức token, nhưng RM chỉ chấm điểm toàn bộ câu trả lời.

**Tại sao cần KL?** Không có nó, PPO sẽ vui vẻ tìm ra các chiến lược reward hacking (tấn công phần thưởng) — RM chỉ được huấn luyện trên các completion trong phân phối. Một câu trả lời ngoài phân phối có thể được điểm cao hơn bất kỳ câu nào con người viết. KL giữ `π_θ` gần đa tạp nơi RM được huấn luyện. Đây là núm vặn quan trọng nhất trong RLHF.

**Tình hình 2026:**

- **DPO** (Rafailov 2023): đại số dạng đóng gộp Giai đoạn 2+3 thành một loss supervised duy nhất trên dữ liệu sở thích. Không RM, không PPO. Chất lượng tương đương trên benchmark alignment với một phần nhỏ compute. Được trình bày ở Giai đoạn 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO với baseline tương đối theo nhóm thay cho critic, phần thưởng đến từ một *verifier* (code chạy được / đáp án toán khớp) thay vì RM do con người huấn luyện. Thống trị cho mô hình suy luận. Được trình bày ở Giai đoạn 9 · 12.
- **Process reward models (PRMs):** chấm điểm các lời giải từng phần (mỗi bước suy luận), được dùng trong cả RLHF và các biến thể GRPO cho suy luận.
- **Constitutional AI / RLAIF:** dùng một LLM đã alignment để tạo sở thích thay vì con người. Mở rộng ngân sách sở thích.

```figure
reward-model
```

## Tự tay làm

Bài này dùng các "prompt" và "câu trả lời" tổng hợp tí hon được biểu diễn bằng chuỗi. RM là một bộ chấm điểm tuyến tính trên biểu diễn túi token (bag-of-tokens). Không có LLM thật — *hình dạng* của pipeline mới quan trọng, không phải quy mô. Xem `code/main.py`.

### Bước 1: dữ liệu sở thích tổng hợp

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

Trong RLHF thật, phần này được thay bằng những người gán nhãn. Hình dạng — `(prompt, preferred_response, rejected_response)` — là y hệt.

### Bước 2: mô hình phần thưởng Bradley-Terry

Điểm tuyến tính: `R(x, y) = w · bag(y)`. Huấn luyện để cực tiểu log-loss so sánh theo cặp của BT:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Sau vài trăm vòng cập nhật, `w` gán trọng số dương cho các token từ tốt và trọng số âm cho từ xấu.

### Bước 3: policy kiểu PPO trên RM

Policy đồ chơi của chúng ta sinh ra một token duy nhất từ một bộ từ vựng. Ta chấm điểm token đó dưới RM, tính `log π_θ(token | prompt)`, cộng thêm hình phạt KL so với reference, và áp dụng surrogate PPO có cắt.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Bước 4: theo dõi KL

Theo dõi `KL(π_θ || π_ref)` trung bình sau mỗi vòng cập nhật. Nếu nó vượt `~5-10` thì policy đã trôi xa khỏi `π_SFT` — có nghĩa là cần hạ `β` hoặc reward hacking đang bắt đầu. Đây là chẩn đoán hàng đầu trong RLHF thật.

### Bước 5: công thức production với TRL

Khi đã hiểu pipeline đồ chơi, đây là cùng vòng lặp đó như một người dùng thư viện thật sẽ viết. [TRL](https://huggingface.co/docs/trl) của Hugging Face là bản triển khai tham chiếu — `RewardTrainer` cho Giai đoạn 2 và `PPOTrainer` (đã tích hợp KL so với reference) cho Giai đoạn 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Thư viện làm ba việc thay bạn. `adap_kl_ctrl=True` cài đặt lịch β thích nghi: nếu KL quan sát vượt `target_kl` thì β nhân đôi; nếu dưới một nửa thì β giảm một nửa. Mô hình reference được đóng băng theo quy ước — bạn không được vô tình chia sẻ tham số với `policy`. Và value head nằm trên cùng backbone với policy (`AutoModelForCausalLMWithValueHead` gắn thêm một head MLP vô hướng), đó là lý do TRL báo cáo `policy/kl` và `value/loss` riêng.

## Cạm bẫy

- **Tối ưu quá mức / reward hacking.** RM là không hoàn hảo; `π_θ` tìm ra các completion đối nghịch được điểm cao nhưng lại dở. Triệu chứng: phần thưởng leo mãi trong khi điểm đánh giá của con người đi ngang hoặc giảm. Cách sửa: dừng sớm, tăng `β`, mở rộng dữ liệu huấn luyện RM.
- **Hack độ dài.** Các RM huấn luyện trên câu trả lời hữu ích thường ngầm thưởng cho độ dài. Policy học cách kéo dài câu trả lời. Cách khắc phục: chuẩn hóa phần thưởng theo độ dài, hoặc dùng RLAIF với một RM nhận biết độ dài.
- **RM quá nhỏ.** RM cần ít nhất lớn bằng policy. Một RM tí hon không thể chấm điểm trung thực đầu ra của policy.
- **Tinh chỉnh KL.** β quá thấp → trôi và reward hacking. β quá cao → policy gần như không đổi. Mẹo tiêu chuẩn là β *thích nghi* nhắm tới một mức KL cố định mỗi bước.
- **Nhiễu trong dữ liệu sở thích.** Khoảng 30% nhãn của con người là nhiễu hoặc mơ hồ. Hiệu chỉnh bằng cách huấn luyện RM trên dữ liệu đã lọc theo mức đồng thuận hoặc dùng temperature trên BT.
- **Vấn đề off-policy.** Dữ liệu PPO hơi off-policy sau epoch đầu. Theo dõi tỷ lệ cắt như ở Bài 08.

## Áp dụng

RLHF năm 2026 được chia tầng:

| Tầng | Mục tiêu | Cách làm |
|-------|--------|--------|
| Tuân theo chỉ dẫn, hữu ích, không gây hại | Alignment | DPO (Giai đoạn 10 · 08) được ưu tiên hơn RLHF-PPO. |
| Tính đúng của suy luận (toán, code) | Năng lực | GRPO với phần thưởng từ verifier (Giai đoạn 9 · 12). |
| Tác vụ nhiều bước, tầm xa | Agentic | PPO / GRPO với process reward model trên từng bước. |
| An toàn / hành vi từ chối | An toàn | RLHF-PPO với RM an toàn riêng, hoặc Constitutional AI. |
| Best-of-N lúc suy luận | Alignment nhanh | Dùng RM ở lúc decode; không cần huấn luyện policy. |
| Chưng cất phần thưởng | Compute lúc suy luận | Huấn luyện một "reward head" nhỏ trên LM đóng băng. |

RLHF là *phương pháp* của giai đoạn 2022–2024. Năm 2026, các pipeline alignment trong production ưu tiên DPO, chỉ dùng PPO cho những bước nặng về RM hoặc quan trọng về an toàn.

## Đóng gói

Lưu thành `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Bài tập

1. **Dễ.** Huấn luyện mô hình phần thưởng Bradley-Terry trong `code/main.py` trên 500 cặp sở thích tổng hợp. Đo độ chính xác so sánh theo cặp trên 100 cặp giữ riêng. Nên vượt 90%.
2. **Trung bình.** Chạy vòng lặp PPO-RLHF đồ chơi với `β ∈ {0.0, 0.1, 1.0}`. Với mỗi giá trị, vẽ đồ thị điểm RM so với KL so với reference qua các vòng cập nhật. Cấu hình nào bị reward hack?
3. **Khó.** Cài đặt DPO (loss hợp lý của sở thích dạng đóng) trên cùng dữ liệu sở thích và so sánh với pipeline RLHF-PPO về compute đã dùng và điểm RM cuối cùng đạt được.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| RLHF | "RL cho alignment" | Pipeline ba giai đoạn SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "Mạng chấm điểm" | Hàm vô hướng học được, khớp với sở thích theo cặp qua Bradley-Terry. |
| Bradley-Terry | "Loss logistic theo cặp" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; mục tiêu chuẩn của RM. |
| KL penalty | "Bám gần reference" | `β · KL(π_θ \|\| π_ref)` trong phần thưởng; bộ điều chuẩn chống reward hacking. |
| Reward hacking | "Định luật Goodhart" | Policy khai thác lỗ hổng của RM; triệu chứng: phần thưởng tăng, đánh giá của con người đi ngang. |
| RLAIF | "Sở thích do AI gán nhãn" | RLHF trong đó nhãn đến từ một LM khác thay vì con người. |
| PRM | "Process Reward Model" | Chấm điểm các bước suy luận từng phần; dùng trong pipeline suy luận. |
| Constitutional AI | "Phương pháp của Anthropic" | Sở thích do AI tạo ra, được hướng dẫn bởi các quy tắc tường minh. |

## Đọc thêm

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — bài báo khởi đầu RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — công thức đằng sau ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — RLHF thời kỳ đầu cho tóm tắt văn bản.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; lựa chọn mặc định hậu RLHF trong 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF và vòng tự phê bình.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — bài báo HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — `RewardTrainer` và `PPOTrainer` cho production. Đọc mã nguồn của trainer để hiểu chi tiết KL thích nghi và value head.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) của Lambert, Castricato, von Werra, Havrilla — bài dẫn giải kinh điển về pipeline ba giai đoạn kèm sơ đồ.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — thư viện này; thư mục `examples/` có các script RLHF đầu-cuối cho Llama, Mistral và Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — góc nhìn giả thuyết phần thưởng; tiền đề thiết yếu để nghĩ về reward hacking.

---

[← Bài trước: Tối ưu hoá chính sách lân cận (Proximal Policy Optimization, PPO)](../../08-ppo/docs/vi.md) · [Bài sau: RL đa tác tử →](../../10-multi-agent-rl/docs/vi.md)
