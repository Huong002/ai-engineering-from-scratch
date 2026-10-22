# Gán nhãn từ loại và phân tích cú pháp

> Ngữ pháp có một thời gian dài không được ưa chuộng. Rồi mọi pipeline LLM đều cần kiểm chứng việc trích xuất có cấu trúc, và nó quay trở lại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 01 (Xử lý văn bản), Giai đoạn 2, Bài 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề

Bài 01 đã hứa rằng lemmatization cần một nhãn từ loại. Không biết `running` là động từ thì bộ lemmatize không thể rút gọn nó về `run`. Không biết `better` là tính từ thì nó không thể rút về `good`.

Lời hứa đó che khuất cả một lĩnh vực con. Part-of-speech (từ loại) gán các phạm trù ngữ pháp. Phân tích cú pháp khôi phục cấu trúc cây của câu: từ nào bổ nghĩa cho từ nào, động từ nào chi phối tham tố nào. NLP cổ điển mất hai mươi năm tinh chỉnh cả hai. Rồi học sâu gộp chúng thành một bài toán phân loại token trên nền transformer đã huấn luyện trước, và cộng đồng nghiên cứu chuyển hướng.

Cộng đồng ứng dụng thì không. Mọi pipeline trích xuất có cấu trúc vẫn dùng POS và cây phụ thuộc ở bên dưới. JSON do LLM sinh ra được kiểm chứng với các ràng buộc ngữ pháp. Hệ hỏi đáp phân rã truy vấn nhờ phân tích phụ thuộc. Bộ đánh giá chất lượng dịch máy kiểm tra sự thẳng hàng của cây cú pháp (parse tree).

Đáng để biết. Bài này giới thiệu các bộ nhãn, các mô hình cơ sở, và điểm mà bạn dừng việc tự viết từ đầu để gọi spaCy.

## Khái niệm

**POS tagging** gán cho mỗi token một phạm trù ngữ pháp, tức một nhãn từ loại (POS tag). Bộ nhãn **Penn Treebank (PTB)** là mặc định cho tiếng Anh. 36 nhãn với những phân biệt mà người đọc thông thường thấy rườm rà: `NN` danh từ số ít, `NNS` danh từ số nhiều, `NNP` danh từ riêng số ít, `VBD` động từ quá khứ, `VBZ` động từ ngôi thứ 3 số ít hiện tại, v.v. Bộ nhãn **Universal Dependencies (UD)** thô hơn (17 nhãn) và không phụ thuộc ngôn ngữ; nó trở thành mặc định cho công việc đa ngữ.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Phân tích cú pháp** tạo ra một cây. Hai trường phái chính:

- **Constituency parsing (phân tích thành phần).** Các cụm danh từ, cụm động từ, cụm giới từ lồng vào nhau. Đầu ra là một cây các phạm trù không kết thúc (NP, VP, PP) với từ là lá.
- **Dependency parsing (phân tích phụ thuộc).** Mỗi từ có một từ chủ duy nhất mà nó phụ thuộc vào, được gán nhãn bằng một quan hệ ngữ pháp. Đầu ra là một cây mà mỗi cạnh là một bộ ba (head, dependent, relation).

Phân tích phụ thuộc thắng trong những năm 2010 vì nó tổng quát hoá gọn gàng qua các ngôn ngữ, đặc biệt là những ngôn ngữ có trật tự từ tự do.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## Tự tay làm

### Bước 1: baseline nhãn thường gặp nhất

Bộ gán nhãn (tagger) POS ngốc nghếch nhất mà vẫn chạy được. Với mỗi từ, dự đoán nhãn mà nó mang thường xuyên nhất trong tập huấn luyện.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Trên kho ngữ liệu Brown, baseline này đạt độ chính xác ~85%. Không tốt, nhưng là mức sàn mà không mô hình nghiêm túc nào được phép xuống dưới.

### Bước 2: bộ gán nhãn HMM bigram

Mô hình hoá xác suất đồng thời của chuỗi:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Hai bảng: xác suất chuyển (nhãn khi biết nhãn trước), xác suất phát (từ khi biết nhãn). Ước lượng cả hai từ số đếm với làm trơn Laplace. Giải mã bằng Viterbi (quy hoạch động trên lưới nhãn).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM trên Brown đạt độ chính xác ~93%. Bước nhảy từ 85% lên 93% chủ yếu nhờ xác suất chuyển — mô hình học được rằng `DET NOUN` là phổ biến còn `NOUN DET` là hiếm.

### Bước 3: vì sao các bộ gán nhãn hiện đại vượt qua mô hình này

Xác suất chuyển và xác suất phát đều mang tính cục bộ. Chúng không nắm được rằng `saw` là danh từ trong "I bought a saw" nhưng là động từ trong "I saw the movie." Một CRF với các đặc trưng tuỳ ý (hậu tố, hình dạng từ, từ trước và từ sau, chính từ đó) đạt ~97%. Một BiLSTM-CRF hay transformer đạt ~98%+.

Trần của bài toán này do sự bất đồng giữa những người gán nhãn đặt ra. Con người đồng thuận khoảng 97% số lần trên Penn Treebank. Các mô hình vượt 98% có lẽ đang overfit tập kiểm tra.

### Bước 4: phác thảo phân tích phụ thuộc

Phân tích phụ thuộc đầy đủ từ đầu nằm ngoài phạm vi; cách trình bày kinh điển của sách giáo khoa là ở Jurafsky và Martin. Hai họ cổ điển cần biết:

- **Bộ phân tích dựa trên chuyển trạng thái (transition-based)** (arc-eager, arc-standard) hoạt động như một bộ phân tích shift-reduce: chúng đọc các token, đẩy chúng vào ngăn xếp, rồi áp dụng các hành động reduce để tạo cung. Giải mã tham lam thì nhanh. Cài đặt kinh điển là MaltParser. Phiên bản neural hiện đại: bộ phân tích chuyển trạng thái của Chen và Manning.
- **Bộ phân tích dựa trên đồ thị (graph-based)** (thuật toán Eisner, Dozat-Manning biaffine) chấm điểm mọi cạnh chủ–phụ có thể rồi chọn cây khung lớn nhất. Chậm hơn nhưng chính xác hơn.

Với hầu hết công việc ứng dụng, hãy gọi spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Đọc cột `dep` từ dưới lên trên là cấu trúc ngữ pháp của câu hiện ra.

## Áp dụng

Mọi thư viện NLP chạy production đều cung cấp bộ gán nhãn POS và bộ phân tích phụ thuộc như một phần của pipeline chuẩn.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Nhanh, chính xác, tích hợp với tokenization + NER + lemmatization. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (quan hệ phụ thuộc).
- **Stanford NLP (stanza)**. Sản phẩm kế nhiệm CoreNLP của Stanford. Đạt mức tốt nhất trên 60+ ngôn ngữ.
- **trankit**. Dựa trên transformer, độ chính xác UD tốt.
- **NLTK**. `pos_tag`. Dùng được, chậm, cũ hơn. Ổn cho việc dạy học.

### Những chỗ việc này vẫn còn quan trọng vào năm 2026

- **Lemmatization.** Bài 01 cần POS để lemmatize đúng. Luôn luôn là vậy.
- **Trích xuất có cấu trúc từ đầu ra của LLM.** Kiểm chứng rằng một câu được sinh ra tuân thủ các ràng buộc ngữ pháp (ví dụ: hoà hợp chủ–vị, các bổ ngữ bắt buộc).
- **Cảm xúc theo khía cạnh.** Phân tích phụ thuộc cho bạn biết tính từ nào bổ nghĩa cho danh từ nào.
- **Hiểu truy vấn.** "movies directed by Wes Anderson starring Bill Murray" được phân rã thành các ràng buộc có cấu trúc nhờ cây phân tích.
- **Chuyển giao đa ngữ.** Nhãn UD và quan hệ phụ thuộc không phụ thuộc ngôn ngữ, cho phép phân tích có cấu trúc zero-shot trên ngôn ngữ mới.
- **Pipeline ít tính toán.** Nếu bạn không thể triển khai một transformer, POS + phân tích phụ thuộc + gazetteer đưa bạn đi xa đến mức đáng ngạc nhiên.

## Đóng gói

Lưu thành `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Bài tập

1. **Dễ.** Dùng baseline nhãn thường gặp nhất trên một kho ngữ liệu gán nhãn nhỏ (ví dụ: tập con Brown của NLTK), đo độ chính xác trên các câu held-out. Kiểm chứng lại kết quả ~85%.
2. **Trung bình.** Huấn luyện bigram HMM ở trên và báo cáo precision/recall theo từng nhãn. Những nhãn nào HMM nhầm lẫn nhiều nhất?
3. **Khó.** Dùng phân tích phụ thuộc của spaCy để trích xuất các bộ ba chủ–động–tân từ một mẫu 1000 câu. Đánh giá trên 50 bộ ba được gán nhãn thủ công. Ghi lại những chỗ việc trích xuất thất bại (thường là câu bị động, cấu trúc đẳng lập, và chủ ngữ bị tỉnh lược).

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| POS tag | Loại của từ | Phạm trù ngữ pháp. PTB có 36; UD có 17. |
| Penn Treebank | Bộ nhãn chuẩn | Riêng cho tiếng Anh. Phân biệt mịn thì của động từ và số của danh từ. |
| Universal Dependencies | Bộ nhãn đa ngữ | Thô hơn PTB; trung tính về ngôn ngữ; là mặc định cho công việc đa ngữ. |
| Dependency parse | Cây của câu | Mỗi từ có một từ chủ, mỗi cạnh có một quan hệ ngữ pháp. |
| Viterbi | Quy hoạch động | Tìm chuỗi nhãn có xác suất cao nhất với xác suất phát và xác suất chuyển đã cho. |

## Đọc thêm

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — cách trình bày kinh điển của sách giáo khoa về POS và phân tích cú pháp.
- [Universal Dependencies project](https://universaldependencies.org/) — bộ nhãn và bộ sưu tập treebank đa ngữ mà mọi bộ phân tích đa ngữ đều dùng.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — tài liệu tham khảo thực dụng cho mọi thuộc tính được phơi ra trên `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — bài báo đưa các bộ phân tích neural vào dòng chính.

---

[← Bài trước: Named Entity Recognition (nhận dạng thực thể có tên)](../../06-named-entity-recognition/docs/vi.md) · [Bài sau: CNNs và RNN cho Văn bản →](../../08-cnns-rnns-for-text/docs/vi.md)
