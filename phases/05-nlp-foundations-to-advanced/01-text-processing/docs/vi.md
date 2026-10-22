# Xử lý văn bản — Tokenization, Stemming, Lemmatization

> Ngôn ngữ là liên tục. Mô hình là rời rạc. Tiền xử lý là cây cầu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 2, Bài 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề

Một mô hình không đọc được "The cats were running." Nó đọc các số nguyên.

Mọi hệ thống NLP đều mở đầu bằng ba câu hỏi giống nhau. Từ bắt đầu ở đâu. Gốc của từ là gì. Làm sao coi "run", "running", "ran" là cùng một thứ khi điều đó có ích, và là những thứ khác nhau khi không có ích.

Làm sai tokenization thì mô hình học từ rác. Nếu bộ tách token coi `don't` là một token nhưng `do n't` là hai, phân phối huấn luyện bị tách đôi. Nếu bộ tách gốc từ gộp `organization` và `organ` về cùng một gốc, mô hình chủ đề chết. Nếu bộ lemmatize cần ngữ cảnh từ loại mà bạn không truyền vào, động từ bị xử lý như danh từ.

Bài này tự tay xây ba bước tiền xử lý, rồi cho thấy NLTK và spaCy làm cùng việc đó ra sao để bạn thấy các đánh đổi.

## Khái niệm

Ba phép toán. Mỗi phép có một nhiệm vụ và một kiểu thất bại.

**Tokenization** (tách token) chia một chuỗi thành các token. "Token" cố tình mơ hồ vì độ chi tiết đúng phụ thuộc vào bài toán. Mức từ cho NLP cổ điển. Mức subword (đơn vị dưới từ) cho transformer. Mức ký tự cho ngôn ngữ không có whitespace (khoảng trắng).

**Stemming** (tách gốc từ) chặt đuôi bằng luật. Nhanh, mạnh tay, ngốc nghếch. `running -> run`. `organization -> organ`. Trường hợp thứ hai chính là kiểu thất bại.

**Lemmatization** (đưa về dạng từ điển) rút một từ về dạng từ điển của nó bằng kiến thức ngữ pháp. Chậm hơn, chính xác hơn, cần bảng tra hoặc bộ phân tích hình thái. `ran -> run` (cần biết "ran" là quá khứ của "run"). `better -> good` (cần biết các dạng so sánh hơn).

Rule of thumb (quy tắc kinh nghiệm). Dùng stemming khi tốc độ quan trọng và bạn chịu được nhiễu (đánh chỉ mục tìm kiếm, phân loại thô). Dùng lemmatization khi ý nghĩa quan trọng (hỏi đáp, tìm kiếm ngữ nghĩa, bất cứ thứ gì người dùng sẽ đọc).

```figure
edit-distance
```

## Tự tay làm

### Bước 1: một bộ tách token bằng regex

Bộ tách token hữu dụng đơn giản nhất là tách theo ký tự không phải chữ/số, đồng thời giữ dấu câu thành token riêng. Không hoàn hảo, chưa phải bản cuối, nhưng chỉ chạy trong một dòng.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Ba mẫu theo thứ tự ưu tiên. Từ có dấu nháy đơn bên trong tuỳ chọn (`don't`, `it's`). Số thuần. Mọi ký tự đơn không phải khoảng trắng và không phải chữ/số, thành token độc lập (dấu câu).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Các kiểu thất bại cần để ý. `3pm` bị tách thành `['3', 'pm']` vì ta xen kẽ giữa cụm chữ và cụm số. Đủ tốt cho hầu hết bài toán. URL, email, hashtag đều hỏng. Với môi trường production, thêm các mẫu riêng trước các mẫu tổng quát.

### Bước 2: một Porter stemmer (chỉ bước 1a)

Thuật toán Porter đầy đủ có năm pha luật. Riêng bước 1a đã bao được các đuôi tiếng Anh phổ biến nhất và dạy được khuôn mẫu.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Đọc các luật từ trên xuống. Luật `ies -> i` là lý do `ponies -> poni`, không phải `pony`. Porter thật có bước 1b sẽ sửa việc đó. Các luật tranh nhau. Luật đứng trước thắng. Thứ tự quan trọng hơn bất kỳ luật riêng lẻ nào.

### Bước 3: một bộ lemmatize dựa trên bảng tra

Lemmatization đúng nghĩa cần hình thái học. Một bản dạy học vừa sức dùng một bảng lemma nhỏ kèm phương án dự phòng.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

Trường hợp cuối là điểm dạy học then chốt. `watched` không có trong bảng của ta và phương án dự phòng chỉ xử lý `ing`. Lemmatization thật bao cả `ed`, động từ bất quy tắc, tính từ so sánh, số nhiều có biến đổi âm (`children -> child`). Đó là lý do hệ thống production dùng WordNet, bộ phân tích hình thái của spaCy, hoặc một bộ phân tích hình thái đầy đủ.

### Bước 4: nối chúng lại

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Mảnh còn thiếu là bộ gán nhãn POS. Giai đoạn 5 · 07 (POS Tagging) xây một bộ. Tạm thời, mặc định mọi thứ là `NOUN` và thừa nhận giới hạn đó.

## Áp dụng

NLTK và spaCy cung cấp bản production. Mỗi bên vài dòng.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` xử lý viết tắt, Unicode, các ca biên mà regex của bạn bỏ sót. `PorterStemmer` chạy đủ năm pha. `WordNetLemmatizer` cần nhãn POS được chuyển từ hệ Penn Treebank của NLTK sang bộ viết tắt của WordNet. Phần dây nối chuyển đổi ở trên là chỗ hầu hết tutorial bỏ qua.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy giấu toàn bộ pipeline sau `nlp(text)`. Tách token, gán nhãn POS, và lemmatization đều chạy. Nhanh hơn NLTK khi quy mô lớn. Chính xác hơn ngay khi cài. Đánh đổi là bạn không dễ thay từng thành phần riêng lẻ.

### Khi nào chọn cái nào

| Tình huống | Chọn |
|-----------|------|
| Dạy học, nghiên cứu, thay đổi từng thành phần | NLTK |
| Production, đa ngôn ngữ, tốc độ quan trọng | spaCy |
| Pipeline transformer (dù sao bạn cũng tách token bằng tokenizer của mô hình) | Dùng `tokenizers` / `transformers` và bỏ qua tiền xử lý cổ điển |

### Hai kiểu thất bại không ai cảnh báo bạn

Hầu hết tutorial dạy thuật toán rồi dừng. Hai thứ sẽ cắn một pipeline tiền xử lý thật, và gần như không bao giờ được nhắc tới.

**Trôi khả năng tái lập.** NLTK và spaCy thay đổi cách tách token và cách lemmatize giữa các phiên bản. Thứ tạo ra `['do', "n't"]` trong spaCy 2.x có thể tạo ra `["don't"]` trong 3.x. Mô hình của bạn được huấn luyện trên một phân phối. Suy luận giờ chạy trên phân phối khác. Độ chính xác âm thầm giảm và không ai biết tại sao. Ghim phiên bản thư viện trong `requirements.txt`. Viết một test hồi quy tiền xử lý chốt kết quả tách token kỳ vọng của 20 câu mẫu. Chạy nó mỗi lần nâng cấp.

**Lệch giữa huấn luyện và suy luận.** Huấn luyện với tiền xử lý mạnh tay (chữ thường, bỏ stopword, stemming), triển khai trên đầu vào thô của người dùng, rồi xem hiệu năng sụp. Đây là lỗi NLP production phổ biến nhất. Nếu bạn tiền xử lý khi huấn luyện, bạn phải chạy đúng hàm đó khi suy luận. Đóng gói tiền xử lý thành một hàm bên trong package của mô hình, không phải một cell notebook để đội serving viết lại.

## Đóng gói

Một prompt tái sử dụng giúp kỹ sư chọn chiến lược tiền xử lý mà không phải đọc ba cuốn sách giáo khoa.

Lưu thành `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Bài tập

1. **Dễ.** Mở rộng `tokenize` để giữ URL thành một token duy nhất. Kiểm tra: `tokenize("Visit https://example.com today.")` phải cho ra một token URL.
2. **Trung bình.** Cài đặt Porter bước 1b. Nếu một từ chứa nguyên âm và kết thúc bằng `ed` hoặc `ing`, bỏ đuôi đó. Xử lý luật phụ âm đôi (`hopping -> hop`, không phải `hopp`).
3. **Khó.** Xây một bộ lemmatize dùng WordNet làm bảng tra nhưng quay về Porter stemmer của bạn khi WordNet không có mục. Đo độ chính xác trên một corpus đã gán nhãn, so với WordNet thuần và Porter thuần.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Token | Một từ | Đơn vị mà mô hình tiêu thụ. Có thể là từ, subword, ký tự, hoặc byte. |
| Stem | Gốc của một từ | Kết quả của việc chặt đuôi theo luật. Không phải lúc nào cũng là một từ thật. |
| Lemma | Dạng từ điển | Dạng bạn sẽ tra. Cần ngữ cảnh ngữ pháp để tính đúng. |
| POS tag | Từ loại | Phạm trù như NOUN, VERB, ADJ. Cần để lemmatize chính xác. |
| Morphology | Luật hình dạng từ | Cách một từ đổi dạng theo thì, số, cách. Lemmatization phụ thuộc vào nó. |

## Đọc thêm

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — bài báo gốc, năm trang, vẫn là lời giải thích rõ nhất.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — cách một pipeline thật được nối dây.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — các ca biên khi tách token mà bạn chưa nghĩ tới.

---

[Bài sau: Bag of Words, TF-IDF, và Biểu diễn Văn bản →](../../02-bag-of-words-tfidf/docs/vi.md)
