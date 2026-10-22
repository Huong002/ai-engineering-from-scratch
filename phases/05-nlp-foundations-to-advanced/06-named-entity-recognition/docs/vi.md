# Named Entity Recognition (nhận dạng thực thể có tên)

> Kéo các cái tên ra ngoài. Nghe dễ cho tới khi bạn gặp ranh giới mơ hồ, thực thể lồng nhau và biệt ngữ chuyên ngành.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 02 (BoW + TF-IDF), Giai đoạn 5, Bài 03 (Word Embeddings)
**Time:** ~75 phút

## Vấn đề

"Apple sued Google over its iPhone search deal in the US." Năm thực thể: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (có thể), US (GPE). Một hệ thống NER tốt trích ra tất cả với đúng loại thực thể. Một hệ thống tồi bỏ sót iPhone, nhầm Apple là quả táo với Apple là công ty, và gán "US" là PERSON.

NER là con ngựa thồ bên dưới mọi pipeline trích xuất có cấu trúc. Phân tích CV, quét log tuân thủ, ẩn danh hồ sơ y tế, hiểu truy vấn tìm kiếm, làm nền cho câu trả lời chatbot, trích xuất hợp đồng pháp lý. Bạn gần như không bao giờ thấy nó; bạn luôn phụ thuộc vào nó.

Bài này đi con đường cổ điển (rule-based, HMM, CRF) rồi sang con đường hiện đại (BiLSTM-CRF, sau đó transformer). Mỗi bước giải quyết một hạn chế cụ thể của bước trước. Chính mẫu hình đó là bài học.

## Khái niệm

**BIO tagging** (gắn nhãn BIO) (hoặc BILOU) biến việc trích xuất thực thể thành bài toán gắn nhãn chuỗi. Gán cho mỗi token một nhãn `B-TYPE` (bắt đầu thực thể), `I-TYPE` (bên trong thực thể), hoặc `O` (ngoài mọi thực thể).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Thực thể nhiều token thì nối lại: `New B-GPE`, `York I-GPE`, `City I-GPE`. Một mô hình hiểu BIO có thể trích ra các span (đoạn thực thể) tuỳ ý.

Diễn tiến kiến trúc:

- **Rule-based.** Regex + tra cứu gazetteer (từ điển tên/địa danh). Độ chính xác cao trên thực thể đã biết, độ phủ bằng không với thực thể mới.
- **HMM.** Hidden Markov Model. Xác suất phát xạ của token với nhãn, xác suất chuyển tiếp từ nhãn sang nhãn. Giải mã Viterbi. Huấn luyện trên dữ liệu đã gán nhãn.
- **CRF.** Conditional Random Field. Như HMM nhưng mang tính phân biệt, nên bạn trộn được đặc trưng tuỳ ý (hình dạng từ, cách viết hoa, từ lân cận). Đến 2026 vẫn là con ngựa thồ cổ điển trong sản xuất cho các triển khai tài nguyên hạn chế.
- **BiLSTM-CRF.** Đặc trưng neural thay cho đặc trưng thủ công. LSTM đọc câu theo cả hai chiều, lớp CRF bên trên ép các chuỗi nhãn nhất quán.
- **Transformer-based.** Fine-tune BERT với một head token classification (phân loại token). Độ chính xác tốt nhất. Tốn compute nhất.

```figure
ner-bio-tagging
```

## Tự tay làm

### Bước 1: Các hàm phụ trợ gắn nhãn BIO

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Bước 2: đặc trưng thủ công

Với NER cổ điển (không neural), đặc trưng là tất cả. Những đặc trưng hữu ích:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` trả về `xXxxxx`. `word_shape("USA-2024")` trả về `XXX-dddd`. Mẫu viết hoa là tín hiệu mạnh cho danh từ riêng.

### Bước 3: baseline rule-based + từ điển đơn giản

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Gazetteer trong sản xuất có hàng triệu mục lấy từ Wikipedia và DBpedia. Độ phủ tốt. Việc phân biệt nghĩa (`Apple` công ty với quả táo) thì rất tệ. Đó là lý do các mô hình thống kê thắng.

### Bước 4: bước CRF (phác thảo, không phải bản đầy đủ)

Tự viết CRF đầy đủ trong 50 dòng không sáng tỏ gì nếu thiếu nền lý thuyết xác suất. Thay vào đó hãy dùng `sklearn-crfsuite`:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` và `c2` là regularization L1 và L2. `all_possible_transitions=True` cho mô hình học rằng những chuỗi không hợp lệ (ví dụ `I-ORG` đứng sau `O`) khó xảy ra, đó là cách CRF ép tính nhất quán BIO mà bạn không phải tự viết ràng buộc.

### Bước 5: BiLSTM-CRF thêm được gì

Đặc trưng trở thành học được. Đầu vào: embedding của token (GloVe hoặc fastText). LSTM đọc từ trái sang phải và từ phải sang trái. Các trạng thái ẩn được nối lại rồi đi qua lớp đầu ra CRF. CRF vẫn ép tính nhất quán của chuỗi nhãn; LSTM thay đặc trưng thủ công bằng đặc trưng học được.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Với lớp CRF, dùng `torchcrf.CRF` (pip install pytorch-crf). Mức cải thiện so với CRF thủ công là đo được nhưng nhỏ hơn bạn tưởng, trừ khi bạn có hàng chục nghìn câu đã gán nhãn.

## Áp dụng

spaCy có sẵn NER chất lượng sản xuất ngay khi cài.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Để ý `iPhone` bị gán `ORG` chứ không phải `PRODUCT` — mô hình nhỏ của spaCy phủ thực thể sản phẩm rất yếu. Mô hình lớn (`en_core_web_lg`) làm tốt hơn. Mô hình transformer (`en_core_web_trf`) còn tốt hơn nữa.

Hugging Face cho NER dựa trên BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` gộp các token B-X, I-X liền nhau thành một span. Không có nó, bạn nhận nhãn ở mức token và phải tự gộp.

### NER dựa trên LLM (lựa chọn của năm 2026)

NER bằng LLM zero-shot và few-shot giờ đã cạnh tranh được với mô hình fine-tune trên nhiều miền, và tốt hơn hẳn khi dữ liệu gán nhãn khan hiếm.

- **Zero-shot prompting.** Đưa cho LLM danh sách loại thực thể và một schema ví dụ. Yêu cầu đầu ra JSON. Chạy được ngay; độ chính xác ở mức trung bình trên miền mới.
- **Prompt kiểu ZeroTuneBio.** Chia bài toán thành trích ứng viên → giải thích nghĩa → phán đoán → kiểm tra lại. Prompt nhiều tầng (không phải one-shot) nâng độ chính xác đáng kể trên NER y sinh. Cùng mẫu đó dùng được cho miền pháp lý, tài chính và khoa học.
- **Prompt động với RAG.** Với mỗi lần suy luận, lấy các ví dụ đã gán nhãn giống nhất từ một tập seed nhỏ có chú thích; dựng prompt few-shot ngay tại chỗ. Trong các benchmark 2026, cách này nâng F1 của GPT-4 trên NER y sinh thêm 11-12% so với prompt tĩnh.
- **Chia theo từng loại thực thể.** Với tài liệu dài, một lần gọi trích tất cả loại thực thể cùng lúc sẽ mất recall (độ phủ) khi độ dài tăng. Chạy một lượt trích cho mỗi loại thực thể. Chi phí suy luận cao hơn, độ chính xác cao hơn đáng kể. Đây là mẫu chuẩn cho ghi chú lâm sàng và hợp đồng pháp lý.

Khuyến nghị triển khai tính đến 2026: bắt đầu bằng một baseline LLM zero-shot trước khi thu thập dữ liệu huấn luyện. Thường F1 đã đủ tốt đến mức bạn không bao giờ cần fine-tune.

### Nơi NER cổ điển vẫn thắng

Kể cả khi đã có LLM, NER cổ điển vẫn thắng khi:

- Ngân sách độ trễ dưới 50ms.
- Bạn có hàng nghìn ví dụ đã gán nhãn và cần F1 từ 98% trở lên.
- Miền có ontology ổn định, nơi một CRF hay BiLSTM đã tiền huấn luyện chuyển giao tốt.
- Ràng buộc pháp lý yêu cầu mô hình on-prem, không sinh nội dung.

### Nơi nó sụp đổ

- **Dịch chuyển miền.** NER huấn luyện trên CoNLL khi chạy trên hợp đồng pháp lý lại kém hơn cả một gazetteer. Hãy fine-tune trên miền của bạn.
- **Thực thể lồng nhau.** "Bank of America Tower" đồng thời là ORG và FACILITY. BIO chuẩn không biểu diễn được các span chồng lấn. Bạn cần nested NER (nhiều lượt hoặc mô hình dựa trên span).
- **Thực thể dài.** "United States Federal Deposit Insurance Corporation." Mô hình mức token đôi khi tách đôi chỗ này. Dùng `aggregation_strategy` hoặc hậu xử lý.
- **Loại hiếm.** Nhãn NER y tế như DRUG_BRAND, ADVERSE_EVENT, DOSE. Mô hình đa dụng không biết gì. Scispacy và BioBERT là điểm khởi đầu ở đó.

## Đóng gói

Lưu thành `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Bài tập

1. **Dễ.** Viết `bio_to_spans` (nghịch đảo của `spans_to_bio`) và kiểm tra tính nhất quán khứ hồi trên 10 câu.
2. **Trung bình.** Huấn luyện CRF sklearn-crfsuite ở trên trên bộ dữ liệu NER tiếng Anh CoNLL-2003. Báo F1 theo từng loại thực thể bằng `seqeval`. Kết quả thường thấy: ~84 F1.
3. **Khó.** Fine-tune `distilbert-base-cased` trên một bộ dữ liệu NER chuyên ngành (y tế, pháp lý hoặc tài chính). So sánh với mô hình nhỏ của spaCy. Ghi lại các kiểm tra rò rỉ dữ liệu và viết ra điều làm bạn bất ngờ.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| NER | Trích tên riêng | Gán nhãn các đoạn token theo loại (PERSON, ORG, GPE, DATE, ...). |
| BIO | Sơ đồ gắn nhãn | `B-X` bắt đầu, `I-X` tiếp nối, `O` ở ngoài. |
| BILOU | BIO tốt hơn | Thêm `L-X` (cuối) và `U-X` (đơn vị) cho ranh giới thực thể sạch hơn. |
| CRF | Bộ phân loại có cấu trúc | Mô hình hoá chuyển tiếp giữa các nhãn, không chỉ phát xạ. Ép chuỗi hợp lệ. |
| Nested NER | Thực thể chồng lấn | Một span là thực thể khác với một span con của nó. BIO không diễn đạt được. |
| Entity-level F1 | Thước đo NER đúng | Span dự đoán phải khớp chính xác span thật. F1 mức token thổi phồng độ chính xác. |

## Đọc thêm

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — bài báo BiLSTM-CRF. Kinh điển.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — giới thiệu mẫu token classification đã trở thành chuẩn mực.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — tài liệu thực hành cho mọi thuộc tính trên `Doc.ents` và `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — thư viện đo đúng. Luôn dùng nó.

---

[← Bài trước: Phân tích cảm xúc](../../05-sentiment-analysis/docs/vi.md) · [Bài sau: Gán nhãn từ loại và phân tích cú pháp →](../../07-pos-tagging-parsing/docs/vi.md)
