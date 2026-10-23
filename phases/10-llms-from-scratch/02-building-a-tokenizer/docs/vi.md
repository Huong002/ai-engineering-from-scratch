# Tự xây Tokenizer từ đầu

> Bài 01 cho bạn một món đồ chơi. Bài này cho bạn một vũ khí.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 10, Bài 01 (Tokenizer: BPE, WordPiece, SentencePiece)
**Time:** ~90 phút

## Mục tiêu học tập

- Xây một tokenizer BPE cấp production, xử lý được Unicode, chuẩn hoá khoảng trắng và token đặc biệt
- Cài đặt byte fallback (dự phòng ở mức byte) để tokenizer mã hoá được mọi đầu vào (kể cả emoji, CJK và code) mà không sinh ra token không xác định
- Thêm các mẫu regex tách sơ bộ (pre-tokenization) để chia văn bản tại ranh giới từ trước khi áp dụng các phép gộp (merge) của BPE
- Train (huấn luyện) một tokenizer tuỳ chỉnh trên một corpus và đánh giá tỉ lệ nén (compression ratio) của nó so với tiktoken trên văn bản đa ngôn ngữ

## Vấn đề

Tokenizer BPE của bạn từ Bài 01 chạy tốt với văn bản tiếng Anh. Giờ ném tiếng Nhật vào. Hoặc emoji. Hoặc code Python lẫn tab và dấu cách.

Nó vỡ.

Không phải vì BPE sai -- mà vì bản cài đặt còn thiếu. Một tokenizer production xử lý byte thô ở mọi bảng mã, chuẩn hoá Unicode trước khi chia, quản lý các token đặc biệt không bao giờ bị gộp, nối tách sơ bộ với chia subword, và làm tất cả đủ nhanh để không thành nút thắt của pipeline huấn luyện xử lý 15 nghìn tỉ token.

GPT-2 có 50,257 token. Llama 3 có 128,256. GPT-4 có khoảng 100,000. Đây không phải những con số đồ chơi. Các bảng gộp đứng sau những kích thước từ vựng (vocabulary size) đó được huấn luyện trên hàng trăm gigabyte văn bản, và phần máy móc xung quanh -- chuẩn hoá, tách sơ bộ, chèn token đặc biệt, định dạng chat template -- chính là thứ phân biệt một tokenizer xử lý được "hello world" với một tokenizer xử lý được cả internet.

Bạn sẽ tự xây cỗ máy đó.

## Khái niệm

### Toàn bộ pipeline

Một tokenizer production không phải một thuật toán. Nó là một pipeline gồm năm giai đoạn, mỗi giai đoạn giải một bài toán khác nhau.

```mermaid
graph LR
    A[Văn bản thô] --> B[Chuẩn hoá]
    B --> C[Tách sơ bộ]
    C --> D[Gộp BPE]
    D --> E[Token đặc biệt]
    E --> F[ID token]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Mỗi giai đoạn có một việc riêng:

| Giai đoạn | Việc nó làm | Vì sao quan trọng |
|-------|-------------|----------------|
| Normalize | NFKC Unicode, tuỳ chọn chữ thường, tuỳ chọn bỏ dấu | Ligature "fi" (U+FB01) thành "fi" (hai ký tự). Không có bước này, cùng một từ sẽ ra token khác nhau. |
| Pre-Tokenize | Chia văn bản thành khối trước khi BPE chạy | Ngăn BPE gộp vắt qua ranh giới từ. "the cat" không bao giờ được sinh ra token "e c". |
| BPE Merge | Áp dụng các luật gộp đã học lên chuỗi byte | Đây là phần nén cốt lõi. Biến byte thô thành token subword. |
| Special Tokens | Chèn [BOS], [EOS], [PAD], dấu của chat template | Những token này có ID cố định. Chúng không bao giờ tham gia vào phép gộp BPE. Mô hình cần chúng để có cấu trúc. |
| ID Mapping | Chuyển chuỗi token thành ID số nguyên | Mô hình thấy số nguyên, không thấy chuỗi. |

### BPE ở mức byte

Tokenizer của Bài 01 chạy trên byte UTF-8. Đó là lựa chọn đúng. Nhưng ta đã bỏ qua một điều quan trọng: chuyện gì xảy ra khi những byte đó không phải UTF-8 hợp lệ?

BPE mức byte giải quyết việc này bằng cách coi mọi giá trị byte có thể (0-255) là một token hợp lệ. Bộ từ vựng gốc của bạn đúng 256 mục. Mọi tệp -- văn bản, nhị phân, bị hỏng -- đều tokenize được mà không sinh ra token lạ.

GPT-2 thêm một mẹo: ánh xạ mỗi byte sang một ký tự Unicode in được để bộ từ vựng vẫn đọc được bằng mắt người. Byte 0x20 (dấu cách) thành ký tự "G" trong ánh xạ của họ. Việc này thuần tuý thẩm mỹ. Thuật toán không quan tâm.

Sức mạnh thật sự: BPE mức byte xử lý được mọi ngôn ngữ trên trái đất. Chữ Hán mỗi ký tự là 3 byte UTF-8. Tiếng Nhật có thể 3-4 byte. Ả Rập, Devanagari, emoji -- tất cả chỉ là chuỗi byte. Thuật toán BPE tìm quy luật trong những chuỗi byte này y hệt cách nó tìm quy luật trong byte ASCII tiếng Anh.

### Tách sơ bộ

Trước khi BPE chạm vào văn bản, bạn cần chia nó thành các khối. Việc này ngăn thuật toán gộp tạo ra token vắt qua ranh giới từ.

GPT-2 dùng một mẫu regex để chia văn bản:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Mẫu này chia tại dạng rút gọn ("don't" thành "don" + "'t"), từ kèm dấu cách đầu tuỳ chọn, số, dấu câu và khoảng trắng. Dấu cách đầu được giữ dính vào từ -- nên "the cat" thành [" the", " cat"], chứ không phải ["the", " ", "cat"].

Llama dùng SentencePiece, bỏ qua regex hoàn toàn. Nó coi luồng byte thô là một chuỗi dài và để thuật toán BPE tự tìm ranh giới. Cách này đơn giản hơn nhưng cho BPE nhiều tự do hơn để tạo token vắt qua từ.

Lựa chọn này quan trọng. Regex của GPT-2 ngăn tokenizer học rằng "the" ở cuối một từ và "the" ở đầu từ tiếp theo nên gộp lại. SentencePiece cho phép điều đó, đôi khi tạo ra nén hiệu quả hơn nhưng token khó hiểu hơn.

### Token đặc biệt

Mọi tokenizer production đều dành sẵn ID token cho các dấu cấu trúc:

| Token | Mục đích | Dùng bởi |
|-------|---------|---------|
| `[BOS]` / `<s>` | Bắt đầu chuỗi | Llama 3, GPT |
| `[EOS]` / `</s>` | Kết thúc chuỗi | Tất cả mô hình |
| `[PAD]` | Đệm để căn lô | BERT, T5 |
| `[UNK]` | Token không xác định (BPE mức byte loại bỏ nó) | BERT, WordPiece |
| `<\|im_start\|>` | Bắt đầu ranh giới tin nhắn chat | ChatGPT, Qwen |
| `<\|im_end\|>` | Kết thúc ranh giới tin nhắn chat | ChatGPT, Qwen |
| `<\|user\|>` | Dấu lượt người dùng | Llama 3 |
| `<\|assistant\|>` | Dấu lượt trợ lý | Llama 3 |

Token đặc biệt không bao giờ bị BPE tách. Chúng được so khớp chính xác trước khi thuật toán gộp chạy, thay bằng ID cố định, và văn bản xung quanh được tokenize bình thường.

### Chat template

Đây là chỗ nhiều người nhầm nhất và nhiều bản cài đặt vỡ nhất.

Khi bạn gửi tin nhắn tới một mô hình chat, API nhận một danh sách tin nhắn:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Mô hình không thấy JSON. Nó thấy một chuỗi token phẳng. Chat template chuyển tin nhắn thành chuỗi phẳng đó bằng các token đặc biệt. Mỗi mô hình làm một kiểu khác nhau:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Sai template là mô hình sinh ra rác. Nó được huấn luyện trên đúng một định dạng. Mọi lệch -- thiếu dấu xuống dòng, đổi token, thêm dấu cách -- đẩy đầu vào ra khỏi phân bố huấn luyện.

### Tốc độ

Python quá chậm cho tokenize production.

tiktoken (OpenAI) viết bằng Rust với binding Python. HuggingFace tokenizers cũng bằng Rust. SentencePiece bằng C++. Những thư viện này đạt tốc độ nhanh hơn Python thuần 10-100x.

Để dễ hình dung: tokenize 15 nghìn tỉ token cho tiền huấn luyện Llama 3 ở 1 triệu token mỗi giây (Python nhanh) sẽ mất 174 ngày. Ở 100 triệu token mỗi giây (Rust), mất 1.7 ngày.

Bạn xây bằng Python để hiểu thuật toán. Trong production, bạn sẽ dùng bản cài đặt đã biên dịch và chỉ chạm vào lớp wrapper Python.

```figure
weight-tying
```

## Tự tay làm

### Bước 1: Mã hoá ở mức byte

Nền tảng. Chuyển bất kỳ chuỗi nào thành dãy byte, ánh xạ mỗi byte sang một ký tự in được để hiển thị, rồi đảo ngược lại -- tức khứ hồi (round-trip) về văn bản.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Thử trên văn bản đa ngôn ngữ để thấy số byte:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" là 5 byte. "你好" là 6 byte (3 byte mỗi ký tự). Emoji lửa là 4 byte. Tokenizer mức byte không quan tâm ngôn ngữ gì. Byte là byte.

### Bước 2: Bộ tách sơ bộ bằng Regex

Chia văn bản thành các khối bằng mẫu regex của GPT-2. Mỗi khối được BPE tokenize độc lập.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Module `regex` hỗ trợ escape thuộc tính Unicode (`\p{L}` cho chữ, `\p{N}` cho số). Module `re` của thư viện chuẩn thì không, nên ta dự phòng bằng lớp ký tự ASCII. Với tokenizer đa ngôn ngữ production, hãy cài `regex`.

Thử xem:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Dấu cách đầu vẫn dính vào từ. Dạng rút gọn tách tại dấu nháy đơn. Dấu câu thành khối riêng. BPE sẽ không bao giờ gộp token vắt qua các ranh giới này.

### Bước 3: BPE trên chuỗi byte

Thuật toán cốt lõi từ Bài 01, nhưng giờ chạy độc lập trên từng khối đã tách sơ bộ.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Bước 4: Xử lý token đặc biệt

Token đặc biệt cần so khớp chính xác và ID cố định. Chúng đi vòng qua BPE hoàn toàn.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Bước 5: Lớp Tokenizer đầy đủ

Ghép mọi thứ lại: chuẩn hoá, tách theo token đặc biệt, tách sơ bộ, gộp BPE, ánh xạ sang ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Bước 6: Kiểm thử đa ngôn ngữ

Bài kiểm tra thật. Ném tiếng Anh, tiếng Trung, emoji và code vào.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Chữ Hán cho 3 byte mỗi ký tự. Emoji cho 4 byte. Không cái nào làm tokenizer vỡ. Không cái nào sinh ra token lạ. Đó là sức mạnh của BPE mức byte.

## Áp dụng

### So sánh các tokenizer thật

Nạp tokenizer thật từ Llama 3, GPT-4 và Mistral. Xem mỗi cái xử lý cùng một đoạn văn đa ngôn ngữ ra sao.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Bạn sẽ thấy số token khác nhau cho cùng một văn bản. Llama 3 với bộ từ vựng 128K gộp các mẫu phổ biến mạnh tay hơn. GPT-4 với 100K nằm ở giữa. Mistral với 32K sinh nhiều token hơn nhưng có lớp embedding nhỏ hơn.

Đánh đổi luôn như nhau: bộ từ vựng lớn hơn nghĩa là chuỗi ngắn hơn nhưng nhiều tham số hơn.

## Đóng gói

Bài này sinh ra một prompt để xây và gỡ lỗi tokenizer production. Xem `outputs/prompt-tokenizer-builder.md`.

## Bài tập

1. **Dễ:** Thêm phương thức `get_token_bytes(id)` hiển thị byte thô của bất kỳ ID token nào. Dùng nó để xem các token gộp phổ biến nhất của bạn thực ra đại diện cho cái gì.
2. **Trung bình:** Cài bộ tách sơ bộ kiểu Llama, chia tại khoảng trắng và chữ số nhưng giữ dấu cách đầu. So sánh bộ từ vựng của nó với cách dùng regex GPT-2 trên cùng một corpus.
3. **Khó:** Thêm phương thức chat template nhận danh sách tin nhắn `{"role": ..., "content": ...}` và sinh đúng chuỗi token cho định dạng chat Llama 3. Kiểm thử với bản cài đặt HuggingFace.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Byte-level BPE | "Tokenizer chạy trên byte" | BPE với bộ từ vựng gốc 256 giá trị byte -- xử lý mọi đầu vào mà không có token lạ |
| Pre-tokenization | "Chia trước BPE" | Chia bằng regex hoặc theo luật, ngăn BPE gộp vắt qua ranh giới từ |
| NFKC normalization | "Dọn dẹp Unicode" | Phân rã chuẩn tắc rồi tổng hợp tương thích -- ligature "fi" thành "fi", "A" fullwidth thành "A" |
| Chat template | "Cách tin nhắn thành token" | Định dạng chính xác để chuyển danh sách tin nhắn role/content thành một chuỗi token phẳng -- phụ thuộc mô hình và phải khớp định dạng huấn luyện |
| Special tokens | "Token điều khiển" | Các ID token dành riêng, đi vòng qua BPE -- [BOS], [EOS], [PAD], dấu chat -- so khớp chính xác trước khi gộp |
| Fertility | "Token trên mỗi từ" | Tỉ lệ token đầu ra trên số từ đầu vào -- 1.3 với tiếng Anh trong GPT-4, 2-3 với tiếng Hàn, cao hơn nghĩa là lãng phí context |
| tiktoken | "Tokenizer của OpenAI" | Bản cài đặt BPE bằng Rust với binding Python -- nhanh hơn Python thuần 10-100x |
| Merge table | "Bộ từ vựng" | Danh sách có thứ tự các phép gộp cặp byte học được khi huấn luyện -- đây CHÍNH LÀ tri thức học được của tokenizer |

## Đọc thêm

- [Mã nguồn OpenAI tiktoken](https://github.com/openai/tiktoken) -- bản cài đặt BPE bằng Rust mà GPT-3.5/4 dùng
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- thư viện tokenizer Rust hỗ trợ BPE, WordPiece, Unigram
- [Bài báo Llama 3 (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- chi tiết về bộ từ vựng 128K và huấn luyện tokenizer
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- tokenize không phụ thuộc ngôn ngữ
- [Mã nguồn tokenizer GPT-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- ánh xạ byte sang Unicode gốc

---

[← Bài trước: Tokenizers: BPE, WordPiece, SentencePiece](../../01-tokenizers/docs/vi.md) · [Bài sau: Data pipeline (pipeline dữ liệu) cho Tiền huấn luyện →](../../03-data-pipelines/docs/vi.md)
