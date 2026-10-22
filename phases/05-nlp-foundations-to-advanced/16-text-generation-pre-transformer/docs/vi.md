# Sinh văn bản trước Transformer — Mô hình ngôn ngữ n-gram

> Nếu một từ khiến bạn bất ngờ, mô hình đang tệ. Perplexity (độ hỗn loạn) biến sự bất ngờ thành một con số. Smoothing (làm mượt) giữ cho con số đó hữu hạn.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 01 (Xử lý văn bản), Giai đoạn 2, Bài 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề

Trước transformer, trước RNN, trước word embedding, một language model (mô hình ngôn ngữ) dự đoán từ tiếp theo bằng cách đếm xem từ đó đi sau `n-1` từ trước bao nhiêu lần. Đếm "the cat" → "sat" 47 lần, "the cat" → "jumped" 12 lần, "the cat" → "refrigerator" 0 lần. Chuẩn hoá để có một phân phối xác suất.

Đó chính là mô hình ngôn ngữ n-gram. Nó đã chạy trong mọi bộ nhận dạng giọng nói, mọi bộ kiểm tra chính tả, và mọi hệ dịch máy dựa trên cụm từ từ năm 1980 đến 2015. Nó vẫn chạy khi bạn cần mô hình hoá ngôn ngữ trên thiết bị với chi phí thấp.

Vấn đề thú vị là xử lý những n-gram chưa từng gặp. Một mô hình đếm thô gán xác suất bằng không cho bất cứ thứ gì nó chưa thấy, và đó là thảm hoạ vì câu thì dài mà gần như câu dài nào cũng chứa ít nhất một chuỗi chưa từng gặp. Năm mươi năm nghiên cứu về làm mượt đã giải quyết chuyện đó. Kneser-Ney smoothing là kết quả cuối cùng, và học sâu hiện đại thừa hưởng truyền thống thực nghiệm của nó.

## Khái niệm

![Mô hình n-gram: đếm, làm mượt, sinh](../assets/ngram.svg)

### Trò chơi dự đoán

Trước khi có bất kỳ cỗ máy nào trong số này, một thí nghiệm đã định nghĩa mô hình ngôn ngữ là gì. Che chữ cái tiếp theo của một câu tiếng Anh. Nhờ ai đó đoán nó, mỗi lần một phỏng đoán, cho tới khi đúng. Ghi lại số phỏng đoán. Lặp lại với vài trăm chữ cái.

Số phỏng đoán không phải chuyện vặt. Đó là một cách mã hoá lại văn bản không mất mát: đưa chuỗi số đếm cho một người đoán thứ hai giống hệt, người đó có thể dựng lại từng chữ cái, vì ở mỗi vị trí họ biết chính xác những phỏng đoán nào đến trước. Một thông điệp có thể mã hoá lại bằng ít ký hiệu hơn thì mang ít thông tin hơn trên mỗi ký hiệu, nên thống kê số phỏng đoán đặt một trần cho entropy của tiếng Anh.

Shannon chạy thí nghiệm này năm 1951 và có được một con số vẫn chi phối cả lĩnh vực. Một bảng chữ cái 27 ký hiệu (26 chữ cái cộng dấu cách) có thể mang `log2(27) ≈ 4.75` bit mỗi chữ cái. Người đoán với 100 chữ cái ngữ cảnh đạt từ 0.6 đến 1.3 bit mỗi chữ cái. Tiếng Anh gần như bị ép đi ba phần tư số nước. Cấu trúc mà một mô hình phải học đã được đo trước khi bất kỳ mô hình nào có thể học nó.

Mọi mô hình ngôn ngữ từ đó tới nay đều là một người chơi cơ học của trò chơi này, và mọi con số đánh giá trong bài học này chính là điểm của trò chơi:

- **Cross-entropy loss** là số bit trung bình mô hình cần cho mỗi ký hiệu. Huấn luyện một LM đúng nghĩa là tối thiểu hoá điểm số của nó trong trò chơi đoán.
- **Perplexity** (độ hỗn loạn) là `2^bits` (hoặc `e^nats`): hệ số phân nhánh mà mô hình còn phải đối mặt sau khi đoán xong. Đoán đều trên 27 ký hiệu cho perplexity 27; một người chơi đạt 1 bit mỗi chữ cái có perplexity 2.
- **Độ dài ngữ cảnh là trí nhớ của người chơi.** Một mô hình trigram chơi với hai token trí nhớ. Một transformer chơi đúng trò đó với 100K token. Luật chơi không hề đổi; người chơi giỏi lên.

Có một chỗ đổi đơn vị cần để ý: trò chơi cho điểm theo chữ cái bằng bit (`log2`), còn các công thức n-gram bên dưới cho điểm theo token từ bằng nat (log tự nhiên) — và vì perplexity `e^H` tính bằng nat bằng `2^H` tính bằng bit, hai cách nhìn là cùng một phép đo ở hai đơn vị khác nhau.

```figure
prediction-game
```

**Xác suất n-gram:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Cố định `n` (thường là 3 cho trigram, 4 cho 4-gram). Tính từ số đếm:

```text
P(w | context) = count(context, w) / count(context)
```

**Vấn đề đếm bằng không.** Bất kỳ n-gram nào không thấy trong tập huấn luyện đều nhận xác suất bằng không. Một nghiên cứu năm 2007 trên Brown corpus phát hiện rằng ngay cả một mô hình 4-gram cũng có 30% số 4-gram trong tập held-out là chưa từng thấy khi huấn luyện. Bạn không thể đánh giá trên bất kỳ văn bản thật nào mà không làm mượt.

**Các hướng làm mượt, xếp theo độ tinh vi:**

1. **Laplace (cộng một).** Cộng 1 vào mọi số đếm. Đơn giản, rất tệ với các sự kiện hiếm.
2. **Good-Turing.** Phân bổ lại khối xác suất từ các sự kiện tần suất cao sang các sự kiện chưa thấy, dựa trên tần suất của tần suất.
3. **Interpolation (nội suy).** Kết hợp các ước lượng n-gram, (n-1)-gram, v.v. với trọng số điều chỉnh được.
4. **Backoff.** Nếu n-gram có số đếm bằng không, lùi về (n-1)-gram. Katz backoff chuẩn hoá cách này.
5. **Absolute discounting (chiết khấu tuyệt đối).** Trừ một lượng chiết khấu cố định `D` khỏi mọi số đếm, phân phối lại cho phần chưa thấy.
6. **Kneser-Ney.** Absolute discounting cộng một lựa chọn khéo léo cho mô hình bậc thấp: dùng *continuation probability* (từ này xuất hiện trong bao nhiêu ngữ cảnh khác nhau) thay vì tần suất thô.

Cái nhìn sâu của Kneser-Ney rất đáng chú ý. "San Francisco" là một bigram phổ biến. Unigram "Francisco" hầu như chỉ xuất hiện sau "San." Absolute discounting ngây thơ cho "Francisco" xác suất unigram cao (vì số đếm cao). Kneser-Ney nhận ra "Francisco" chỉ xuất hiện trong một ngữ cảnh và hạ continuation probability của nó xuống tương ứng. Kết quả: một bigram mới lạ kết thúc bằng "Francisco" nhận đúng xác suất thấp mà nó đáng được nhận.

**Đánh giá: perplexity.** Luỹ thừa của log-likelihood âm trung bình trên mỗi từ, tính trên một tập test held-out. Càng thấp càng tốt. Perplexity 100 nghĩa là mô hình bối rối ngang với việc chọn đều trong 100 từ.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## Tự tay làm

### Bước 1: đếm trigram

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Đầu vào là một danh sách các câu đã tokenize. Đầu ra là số đếm n-gram và số đếm ngữ cảnh. `<s>` và `</s>` là biên câu.

### Bước 2: Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Cộng 1 vào mọi số đếm. Làm mượt nhưng phân bổ quá nhiều khối xác suất cho các sự kiện chưa thấy, làm hại cả những sự kiện hiếm nhưng đã biết.

### Bước 3: Kneser-Ney (bigram, có nội suy)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Có ba phần chuyển động. `continuation_prob` nắm bắt câu hỏi "từ này xuất hiện trong bao nhiêu ngữ cảnh khác nhau?" (đó là phát kiến của Kneser-Ney). `lambda_prev` là khối xác suất được giải phóng nhờ chiết khấu, dùng để cân trọng số cho phần backoff. Xác suất cuối cùng là số hạng chính đã chiết khấu cộng số hạng continuation có trọng số.

### Bước 4: sinh văn bản bằng sampling

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Sampling (lấy mẫu) theo tỉ lệ xác suất. Luôn cho ra kết quả khác nhau theo seed. Muốn đầu ra kiểu beam search, hãy chọn argmax ở mỗi bước (greedy) rồi thêm một nút ngẫu nhiên nhỏ (temperature — nhiệt độ).

### Bước 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Càng thấp càng tốt. Với Brown corpus, một mô hình KN 4-gram được tinh chỉnh tốt đạt perplexity khoảng 140. Một transformer LM đạt 15-30 trên cùng tập test. Khoảng cách khoảng 10 lần. Chính khoảng cách đó khiến cả lĩnh vực phải chuyển hướng.

## Áp dụng

- **Giảng dạy NLP cổ điển.** Cách tiếp cận rõ ràng nhất với làm mượt, MLE và perplexity mà bạn có thể có.
- **KenLM.** Thư viện n-gram cho production. Được dùng làm bộ chấm điểm lại trong hệ thống giọng nói và dịch máy nơi độ trễ thấp là quan trọng.
- **Tự động hoàn thành trên thiết bị.** Mô hình trigram trong bàn phím. Vẫn thế.
- **Baseline.** Luôn tính perplexity của một n-gram LM trước khi tuyên bố neural LM của bạn tốt. Nếu transformer của bạn không vượt KN với biên độ lớn, có gì đó sai.

## Đóng gói

Lưu thành `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Bài tập

1. **Dễ.** Huấn luyện một trigram LM trên một corpus Shakespeare 1.000 câu. Sinh 20 câu. Chúng sẽ hợp lý cục bộ nhưng vô nghĩa toàn cục. Đây là bản demo kinh điển.
2. **Trung bình.** Cài đặt perplexity cho mô hình KN của bạn trên một phần Shakespeare held-out. So sánh với Laplace. Bạn sẽ thấy KN cho perplexity thấp hơn 30-50%.
3. **Khó.** Xây một bộ sửa chính tả trigram: cho một từ viết sai và ngữ cảnh của nó, sinh các phương án sửa rồi xếp hạng theo xác suất ngữ cảnh dưới LM. Đánh giá trên Birkbeck spelling corpus (công khai).

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| N-gram | Chuỗi từ | Chuỗi `n` token liên tiếp. |
| Smoothing | Tránh số không | Phân bổ lại khối xác suất để các sự kiện chưa thấy có xác suất khác không. |
| Perplexity | Thước đo chất lượng LM | `exp(-log-prob trung bình)` trên dữ liệu held-out. Càng thấp càng tốt. |
| Backoff | Lùi về ngữ cảnh ngắn hơn | Nếu số đếm trigram bằng không, dùng bigram. Katz backoff hình thức hoá cách này. |
| Kneser-Ney | Cách làm mượt tốt nhất cho n-gram | Absolute discounting + continuation probability cho mô hình bậc thấp. |
| Continuation probability | Đặc trưng của KN | `P(w)` cân theo số ngữ cảnh mà `w` xuất hiện, không theo số đếm thô. |
| Entropy of text | Thông tin trên mỗi ký hiệu | Số bit trung bình cần để mã hoá ký hiệu tiếp theo, biết ngữ cảnh. Ước lượng năm 1951 của Shannon cho tiếng Anh in với tối đa 100 chữ cái ngữ cảnh: 0.6-1.3 bit/chữ cái, đo trước khi bất kỳ mô hình nào tồn tại. |

## Đọc thêm

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — thí nghiệm trò chơi đoán đã định nghĩa mục tiêu mà mọi mô hình ngôn ngữ vẫn đang tối ưu.
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — cách trình bày kinh điển về n-gram LM và làm mượt.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — bài báo đã định đoạt Kneser-Ney là bộ làm mượt n-gram tốt nhất.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — bài báo KN gốc.
- [KenLM](https://kheafield.com/code/kenlm/) — n-gram LM nhanh cho production, vẫn được dùng năm 2026 cho các ứng dụng nhạy cảm độ trễ.

---

[← Bài trước: Mô hình hoá chủ đề](../../15-topic-modeling/docs/vi.md) · [Bài sau: Chatbot →](../../17-chatbots-rule-to-neural/docs/vi.md)
