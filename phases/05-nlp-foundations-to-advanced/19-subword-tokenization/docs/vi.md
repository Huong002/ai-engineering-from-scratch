# Token hoá dưới từ — BPE, WordPiece, Unigram, SentencePiece

> Token hoá theo từ cứng đờ trước những từ chưa gặp. Token hoá theo ký tự làm độ dài chuỗi phình to. Token hoá dưới từ chọn điểm giữa. Mọi LLM hiện đại đều chạy trên một trong số đó.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 01 (Xử lý văn bản), Giai đoạn 5, Bài 04 (GloVe / FastText / Subword)
**Time:** ~60 phút

## Vấn đề

Từ vựng của bạn có 50,000 từ. Người dùng gõ "untokenizable". Tokenizer của bạn trả về `[UNK]`. Mô hình giờ không còn tín hiệu nào về từ đó. Tệ hơn: tài liệu ở phân vị 90 trong kho ngữ liệu của bạn có 40 từ hiếm, nghĩa là mỗi tài liệu mất đi 40 bit thông tin.

Subword tokenization (token hoá dưới từ) giải quyết việc này. Từ phổ biến vẫn là một token. Từ hiếm được tách thành những mảnh có nghĩa: `untokenizable` → `un`, `token`, `izable`. Dữ liệu huấn luyện phủ được mọi thứ vì suy cho cùng chuỗi nào cũng là một dãy byte.

Mọi LLM tiên phong năm 2026 đều chạy trên một trong ba thuật toán (BPE, Unigram, WordPiece), bọc trong một trong ba thư viện (tiktoken, SentencePiece, HF Tokenizers). Bạn không thể đưa một mô hình ngôn ngữ vào sản xuất mà không chọn một cái.

## Khái niệm

![BPE so với Unigram so với WordPiece, từng ký tự một](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Bắt đầu với từ vựng cấp ký tự. Đếm mọi cặp kề nhau. Hợp nhất (merge) cặp xuất hiện nhiều nhất thành một token mới — đây chính là greedy merge (hợp nhất tham lam). Lặp lại cho tới khi đạt kích thước từ vựng mục tiêu. Thuật toán chiếm ưu thế: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.** Vẫn thuật toán đó nhưng chạy trên byte thô (256 token gốc) thay vì ký tự Unicode — đó là byte-level (mức byte). Bảo đảm không bao giờ có token `[UNK]` — mọi dãy byte đều mã hoá được. GPT-2 dùng 50,257 token (256 byte + 50,000 merge + 1 special token (token đặc biệt)).

**Unigram.** Bắt đầu với một từ vựng khổng lồ. Gán cho mỗi token một xác suất unigram. Lặp đi lặp lại việc cắt bỏ những token mà khi bỏ đi thì log-likelihood của kho ngữ liệu tăng ít nhất. Mang tính xác suất khi suy luận: có thể lấy mẫu cách token hoá (hữu ích cho tăng cường dữ liệu bằng subword regularization). Được T5, mBART, ALBERT, XLNet, Gemma dùng.

**WordPiece.** Hợp nhất những cặp làm likelihood của kho ngữ liệu huấn luyện lớn nhất thay vì theo tần suất thô. Được BERT, DistilBERT, ELECTRA dùng.

**SentencePiece vs tiktoken.** SentencePiece là thư viện *huấn luyện* từ vựng (BPE hoặc Unigram) trực tiếp trên văn bản Unicode thô, mã hoá khoảng trắng thành `▁`. tiktoken là *bộ mã hoá* nhanh của OpenAI chạy trên những từ vựng dựng sẵn; nó không huấn luyện.

Quy tắc chọn nhanh:

- **Huấn luyện một từ vựng mới:** SentencePiece (đa ngôn ngữ, không cần pre-tokenization) hoặc HF Tokenizers.
- **Suy luận nhanh trên từ vựng kiểu GPT:** tiktoken (cl100k_base, o200k_base).
- **Cả hai:** HF Tokenizers — một thư viện, vừa huấn luyện vừa phục vụ.

```figure
bpe-merge
```

## Tự tay làm

### Bước 1: BPE from scratch

Xem `code/main.py`. Vòng lặp:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Ba sự thật mà thuật toán này chứa đựng. `</w>` đánh dấu kết thúc từ để "low" (hậu tố) và "lower" (tiền tố) vẫn khác nhau. Trọng số theo tần suất khiến các cặp tần suất cao thắng sớm. Danh sách merge có thứ tự — khi suy luận, các merge được áp dụng đúng theo thứ tự huấn luyện.

### Bước 2: mã hoá bằng những merge đã học

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Cách ngây thơ tốn O(n·|merges|). Các bản triển khai production (tiktoken, HF Tokenizers) dùng tra cứu hạng merge (merge-rank) với hàng đợi ưu tiên và chạy gần như tuyến tính.

### Bước 3: SentencePiece trong thực tế

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Để ý: không cần pre-tokenization, dấu cách được mã hoá thành `▁`, `character_coverage` điều khiển mức độ giữ lại ký tự hiếm so với việc ánh xạ chúng thành `<unk>`.

### Bước 4: tiktoken cho các từ vựng tương thích OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Chỉ mã hoá. Nhanh (backend Rust). Khớp chính xác với cách GPT-4/5 token hoá, phục vụ đếm byte, ước tính chi phí, lập ngân sách cửa sổ ngữ cảnh.

## Những cạm bẫy vẫn còn trong năm 2026

- **Lệch tokenizer (tokenizer drift).** Huấn luyện trên từ vựng A, triển khai với từ vựng B. ID token khác nhau; đầu ra của mô hình thành rác. Kiểm tra hash `tokenizer.json` trong CI.
- **Nhập nhằng khoảng trắng.** BPE của "hello" và " hello" cho ra token khác nhau. Luôn chỉ định rõ `add_special_tokens` và `add_prefix_space`.
- **Huấn luyện thiếu đa ngôn ngữ.** Kho ngữ liệu nặng tiếng Anh sinh ra từ vựng chia các hệ chữ không phải Latin thành số token gấp 5-10 lần. Cùng một prompt tốn gấp 5-10 lần ở tiếng Nhật/Ả Rập trên GPT-3.5. o200k_base đã sửa được một phần.
- **Emoji bị tách.** Một emoji có thể tốn 5 token. Hãy kiểm tra cách xử lý emoji khi lập ngân sách ngữ cảnh.

## Áp dụng

Bộ công cụ năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Huấn luyện mô hình đơn ngữ từ đầu | HF Tokenizers (BPE) |
| Huấn luyện mô hình đa ngôn ngữ | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Phục vụ API tương thích OpenAI | tiktoken (`o200k_base` cho GPT-4+) |
| Từ vựng chuyên ngành (code, toán, protein) | Huấn luyện BPE riêng trên kho ngữ liệu chuyên ngành, gộp với từ vựng gốc |
| Suy luận trên thiết bị biên, mô hình nhỏ | Unigram (từ vựng nhỏ hơn hoạt động tốt hơn) |

Vocabulary size (kích thước từ vựng) là một quyết định về scaling, không phải một hằng số. Ước lượng thô: 32k cho <1B tham số, 50-100k cho 1-10B, 200k+ cho đa ngôn ngữ/mô hình tiên phong.

## Đóng gói

Lưu thành `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Bài tập

1. **Dễ.** Huấn luyện một BPE 500 merge trên kho ngữ liệu tí hon của `code/main.py`. Mã hoá ba từ chưa từng thấy. Bao nhiêu từ cho ra đúng 1 token, bao nhiêu từ cho ra >1 token?
2. **Trung bình.** So sánh số token trên 100 câu Wikipedia tiếng Anh giữa `cl100k_base`, `o200k_base`, và một SentencePiece BPE do bạn huấn luyện với vocab=32k. Báo cáo tỉ lệ nén của từng cái.
3. **Khó.** Huấn luyện cùng kho ngữ liệu đó với BPE, Unigram và WordPiece. Đo độ chính xác hạ nguồn khi dùng từng cái trên một bộ phân loại cảm xúc nhỏ. Lựa chọn đó có làm F1 dịch đi hơn 1 điểm không?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Hợp nhất tham lam các cặp ký tự xuất hiện nhiều nhất cho tới khi đạt kích thước từ vựng mục tiêu. |
| Byte-level BPE | Không bao giờ có token lạ | BPE trên 256 byte thô; GPT-2 / Llama dùng cách này. |
| Unigram | Tokenizer xác suất | Cắt tỉa từ một tập ứng viên lớn dựa trên log-likelihood; T5, Gemma dùng. |
| SentencePiece | Cái lo khoảng trắng | Thư viện huấn luyện BPE/Unigram trên văn bản thô; dấu cách mã hoá thành `▁`. |
| tiktoken | Cái nhanh | Bộ mã hoá BPE chạy trên Rust của OpenAI cho từ vựng dựng sẵn. Không huấn luyện. |
| Merge list | Mấy con số thần kỳ | Danh sách có thứ tự các merge `(a, b) → ab`; khi suy luận áp dụng theo đúng thứ tự. |
| Character coverage | Hiếm đến đâu là quá hiếm? | Phần ký tự trong kho ngữ liệu huấn luyện mà tokenizer phải phủ; thường là ~0.9995. |

## Đọc thêm

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — bài báo BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — bài báo Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — thư viện.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — tài liệu tham chiếu cô đọng.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + danh sách encoding.

---

[← Bài trước: NLP đa ngôn ngữ](../../18-multilingual-nlp/docs/vi.md) · [Bài sau: Đầu ra có cấu trúc & Giải mã có ràng buộc →](../../20-structured-outputs-constrained-decoding/docs/vi.md)
