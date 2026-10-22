# GloVe, FastText, và Embedding Subword

> Word2Vec huấn luyện một embedding cho mỗi từ. GloVe phân rã ma trận đồng xuất hiện (co-occurrence matrix). FastText nhúng từng mảnh. BPE bắc cầu sang transformer.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 03 (Word2Vec from scratch)
**Time:** ~45 phút

## Vấn đề

Word2Vec để lại hai câu hỏi mở.

Đầu tiên, có một hướng nghiên cứu song song phân rã (factorization) trực tiếp ma trận đồng xuất hiện (LSA, HAL) thay vì cập nhật skip-gram trực tuyến. Cách lặp của Word2Vec có tốt hơn về bản chất không, hay khác biệt chỉ là hệ quả của cách hai phương pháp xử lý số đếm? **GloVe** trả lời câu đó: phân rã ma trận với một hàm mất mát được chọn kỹ càng ngang bằng hoặc vượt Word2Vec, mà chi phí huấn luyện lại thấp hơn.

Thứ hai, cả hai phương pháp đều không có cách nào cho những từ nó chưa từng thấy. `Zoomer-approved`, `dogecoin`, bất kỳ danh từ riêng nào mới ra đời tuần trước, mọi dạng biến cách của một gốc từ hiếm. **FastText** khắc phục bằng cách embedding character n-gram (n-gram ký tự): một từ là tổng các mảnh của nó, kể cả hình vị, nên ngay cả từ ngoài từ vựng (out-of-vocabulary) vẫn có một vector hợp lý.

Thứ ba, khi transformer xuất hiện, câu hỏi lại đổi hướng. Từ vựng theo cấp từ chỉ đạt tối đa khoảng một triệu mục; ngôn ngữ thật mở hơn thế. **Byte-pair encoding (BPE)** và họ hàng của nó giải quyết bằng cách học một từ vựng gồm các đơn vị subword (đơn vị dưới từ) thường gặp, phủ được mọi thứ. Mọi tokenizer hiện đại cho mọi LLM hiện đại đều là tokenizer subword.

Bài này đi qua cả ba, rồi giải thích khi nào thì dùng cái nào.

## Khái niệm

**GloVe (Global Vectors).** Xây ma trận đồng xuất hiện từ-từ `X` trong đó `X[i][j]` là số lần từ `j` xuất hiện trong ngữ cảnh của từ `i`. Huấn luyện các vector sao cho `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Đánh trọng số hàm mất mát để các cặp thường gặp không lấn át. Xong.

**FastText.** Một từ là tổng các character n-gram của nó cộng với chính từ đó. `where` trở thành `<wh, whe, her, ere, re>, <where>`. Vector của từ là tổng các vector thành phần ấy. Huấn luyện như Word2Vec. Lợi ích: từ chưa thấy (`whereupon`) được ghép lại từ các n-gram đã biết.

**BPE (Byte-Pair Encoding).** Bắt đầu với từ vựng gồm từng byte (hoặc từng ký tự) riêng lẻ. Đếm mọi cặp liền kề trong corpus. Gộp cặp thường gặp nhất thành một token mới. Lặp `k` vòng. Kết quả: từ vựng `k + 256` token, trong đó các chuỗi thường gặp (`ing`, `tion`, `the`) là token đơn còn các từ hiếm bị chẻ thành những mảnh quen thuộc. Câu nào cũng tokenize được thành một cái gì đó.

```figure
n5-subword-merge
```

## Tự tay làm

### GloVe: phân rã ma trận đồng xuất hiện

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Có hai chỗ chuyển động đáng gọi tên. Hàm đánh trọng số (weighting) `f(x) = (x/x_max)^alpha` giảm trọng số những cặp rất thường gặp (như `(the, and)`) để chúng không lấn át hàm mất mát. Embedding cuối cùng là tổng của bảng `W` (center) và bảng `W_tilde` (context). Cộng cả hai là một mẹo đã công bố, thường cho kết quả tốt hơn chỉ dùng một bảng.

### FastText: embedding nhận biết subword

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Mỗi từ được biểu diễn bằng tập n-gram của nó (thường từ 3 đến 6 ký tự). Embedding của từ là tổng embedding các n-gram của nó. Khi huấn luyện skip-gram, chỗ nào Word2Vec dùng một vector duy nhất thì bạn đưa cái này vào.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Với một từ chưa thấy, bạn vẫn có vector miễn là một phần các n-gram của nó đã biết. `whereupon` chia sẻ `<wh`, `her`, `ere`, và `<where` với `where`, nên hai từ nằm gần nhau.

### BPE: từ vựng subword học được

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Vòng lặp đầu gộp cặp liền kề thường gặp nhất. Sau đủ số vòng, các chuỗi con thường gặp (`low`, `est`, `tion`) trở thành token đơn, còn các từ hiếm bị chẻ ra gọn gàng.

Tokenizer thật của GPT / BERT / T5 học 30k-100k lần gộp. Kết quả: mọi văn bản đều tokenize thành một chuỗi ID đã biết có độ dài bị chặn, không bao giờ còn OOV.

## Áp dụng

Trong thực tế, bạn hiếm khi tự huấn luyện mấy thứ này. Bạn nạp các checkpoint đã huấn luyện sẵn.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Với tokenize subword kiểu BPE trong thời đại transformer:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Tiền tố `Ġ` đánh dấu ranh giới từ (một quy ước của GPT-2). Mọi tokenizer hiện đại đều là một biến thể BPE, WordPiece (BERT), hoặc SentencePiece (T5, LLaMA).

### Khi nào chọn cái nào

| Tình huống | Chọn |
|-----------|------|
| Vector từ đa dụng huấn luyện sẵn, không cần chịu được OOV | GloVe 300d |
| Vector từ đa dụng huấn luyện sẵn, phải xử lý lỗi gõ (typos) / từ mới / ngôn ngữ giàu hình thái | FastText |
| Bất cứ thứ gì đi vào transformer (huấn luyện hay suy luận) | Đúng tokenizer mà model đó phát hành kèm. Không bao giờ thay. |
| Tự huấn luyện language model từ đầu | Huấn luyện tokenizer BPE hoặc SentencePiece trên corpus của bạn trước |
| Phân loại văn bản production với mô hình tuyến tính | Vẫn TF-IDF. Bài 02. |

## Đóng gói

Lưu thành `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Bài tập

1. **Easy.** Chạy `char_ngrams("playing")` và `char_ngrams("played")`. Tính độ trùng Jaccard của hai tập n-gram. Bạn sẽ thấy nhiều mảnh chung đáng kể (`pla`, `lay`, `play`), đó là lý do FastText chuyển giao tốt giữa các biến thể hình thái.
2. **Medium.** Mở rộng `learn_bpe` để theo dõi mức tăng trưởng từ vựng. Vẽ số token trên mỗi ký tự corpus theo số lần gộp. Bạn sẽ thấy nén nhanh lúc đầu, rồi tiệm cận quanh ~2-3 ký tự mỗi token.
3. **Hard.** Huấn luyện BPE 1k lần gộp trên toàn bộ tác phẩm của Shakespeare. So sánh cách tokenize các từ thường gặp với các danh từ riêng hiếm. Đo số token trung bình mỗi từ trước và sau. Viết lại những gì làm bạn bất ngờ.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Co-occurrence matrix | Bảng tần suất từ-từ | `X[i][j]` = số lần từ `j` xuất hiện trong một cửa sổ quanh từ `i`. |
| Subword | Một mảnh của từ | Một character n-gram (FastText) hoặc một token đã học (BPE/WordPiece/SentencePiece). |
| BPE | Byte-pair encoding | Gộp lặp đi lặp lại cặp liền kề thường gặp nhất cho tới khi từ vựng đạt kích thước mục tiêu. |
| OOV | Out of vocabulary | Từ mà mô hình chưa từng thấy. Word2Vec/GloVe thất bại. FastText và BPE xử lý được. |
| Byte-level BPE | BPE trên byte thô | Cách của GPT-2. Từ vựng bắt đầu với 256 byte, nên không gì còn là OOV nữa. |

## Đọc thêm

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — bài báo GloVe, bảy trang, vẫn là cách dẫn giải hàm mất mát hay nhất.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — bài báo đưa BPE vào NLP hiện đại.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE, WordPiece và SentencePiece thực sự khác nhau thế nào trong thực tế.

---

[← Bài trước: Word Embeddings](../../03-word-embeddings-word2vec/docs/vi.md) · [Bài sau: Phân tích cảm xúc →](../../05-sentiment-analysis/docs/vi.md)
