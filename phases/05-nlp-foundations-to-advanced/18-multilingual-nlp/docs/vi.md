# NLP đa ngôn ngữ

> Một mô hình, hơn 100 ngôn ngữ, và không có dữ liệu huấn luyện cho hầu hết chúng. Chuyển giao cross-lingual (xuyên ngôn ngữ) là điều kỳ diệu thực tế của thập niên 2020.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 04 (GloVe, FastText, Subword), Giai đoạn 5, Bài 11 (Dịch máy)
**Time:** ~45 phút

## Vấn đề

Tiếng Anh có hàng tỷ ví dụ gán nhãn. Tiếng Urdu có hàng nghìn. Tiếng Maithili gần như không có gì. Bất kỳ hệ thống NLP thực tế nào phục vụ khán giả toàn cầu cũng phải hoạt động trên phần đuôi dài của các ngôn ngữ, nơi dữ liệu huấn luyện theo từng tác vụ không tồn tại.

Các mô hình multilingual (đa ngôn ngữ) giải quyết chuyện này bằng cách huấn luyện một mô hình trên nhiều ngôn ngữ cùng lúc. Biểu diễn dùng chung cho phép mô hình chuyển những kỹ năng học được ở ngôn ngữ nhiều tài nguyên sang các ngôn ngữ low-resource (ít tài nguyên). Fine-tune mô hình trên dữ liệu phân tích cảm xúc tiếng Anh, và nó đưa ra dự đoán cảm xúc tốt đến bất ngờ trên tiếng Urdu ngay khi chạy. Đó là chuyển giao zero-shot cross-lingual (không mẫu, xuyên ngôn ngữ), và nó đã định hình lại cách NLP được đưa ra thế giới.

Bài này nêu ra những đánh đổi, những mô hình chuẩn mực, và một quyết định khiến các nhóm mới làm việc đa ngôn ngữ vấp ngã: chọn ngôn ngữ nguồn để chuyển giao.

## Khái niệm

![Chuyển giao xuyên ngôn ngữ qua không gian embedding đa ngôn ngữ dùng chung](../assets/multilingual.svg)

**Từ vựng dùng chung.** Mô hình đa ngôn ngữ dùng tokenizer SentencePiece hoặc WordPiece huấn luyện trên văn bản từ tất cả ngôn ngữ đích. Từ vựng là dùng chung: cùng một đơn vị subword biểu diễn cùng một hình vị trên các ngôn ngữ có quan hệ. `anti-` trong tiếng Anh và tiếng Ý nhận cùng một token.

**Biểu diễn dùng chung.** Một transformer tiền huấn luyện trên masked language modeling qua nhiều ngôn ngữ sẽ học được rằng những câu có nghĩa tương tự ở các ngôn ngữ khác nhau tạo ra hidden state tương tự. mBERT, XLM-R và NLLB đều thể hiện điều này. Embedding của "cat" trong tiếng Anh nằm gần "chat" trong tiếng Pháp và "gato" trong tiếng Tây Ban Nha, và embedding của cả câu cũng vậy.

**Chuyển giao zero-shot.** Fine-tune mô hình trên dữ liệu gán nhãn của một ngôn ngữ (thường là tiếng Anh). Khi suy luận, chạy nó trên bất kỳ ngôn ngữ nào khác mà mô hình hỗ trợ. Không cần nhãn của ngôn ngữ đích. Kết quả tốt với các ngôn ngữ có quan hệ loại hình và yếu hơn với những ngôn ngữ cách xa.

**Fine-tune few-shot.** Thêm 100-500 ví dụ gán nhãn trong ngôn ngữ đích. Độ chính xác nhảy lên 95-98% so với mức cơ sở tiếng Anh trên các tác vụ phân loại. Đây là đòn bẩy hiệu quả nhất về chi phí trong NLP đa ngôn ngữ.

## Các mô hình

| Mô hình | Năm | Phạm vi phủ | Ghi chú |
|-------|------|----------|-------|
| mBERT | 2018 | 104 ngôn ngữ | Huấn luyện trên Wikipedia. LM đa ngôn ngữ thực dụng đầu tiên. Yếu trên ngôn ngữ ít tài nguyên. |
| XLM-R | 2019 | 100 ngôn ngữ | Huấn luyện trên CommonCrawl (lớn hơn Wikipedia rất nhiều). Đặt ra mức cơ sở xuyên ngôn ngữ. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 ngôn ngữ | XLM-R với từ vựng 1M token (so với 250k). Tốt hơn trên ngôn ngữ ít tài nguyên. |
| mT5 | 2020 | 101 ngôn ngữ | Kiến trúc T5 cho sinh văn bản đa ngôn ngữ. |
| NLLB-200 | 2022 | 200 ngôn ngữ | Mô hình dịch của Meta; gồm 55 ngôn ngữ ít tài nguyên. |
| BLOOM | 2022 | 46 ngôn ngữ + 13 ngôn ngữ lập trình | LLM mở 176B huấn luyện đa ngôn ngữ. |
| Aya-23 | 2024 | 23 ngôn ngữ | LLM đa ngôn ngữ của Cohere. Mạnh trên tiếng Ả Rập, Hindi, Swahili. |

Chọn theo tình huống sử dụng. Phân loại hợp với XLM-R-base như mặc định hợp lý. Tác vụ sinh cần mT5 hoặc NLLB tùy vào dịch hay sinh mở. Việc kiểu LLM thì đi với Aya-23 hoặc Claude dùng prompt đa ngôn ngữ rõ ràng.

## Quyết định ngôn ngữ nguồn (nghiên cứu 2026)

Hầu hết các nhóm mặc định lấy tiếng Anh làm nguồn để fine-tune. Nghiên cứu gần đây (2026) cho thấy điều này thường sai.

Độ tương đồng ngôn ngữ dự đoán chất lượng chuyển giao tốt hơn kích thước kho ngữ liệu thô. Với các ngôn ngữ đích Slavic, tiếng Đức hoặc tiếng Nga thường thắng tiếng Anh. Với các ngôn ngữ đích Indic, tiếng Hindi thường thắng tiếng Anh. Thước đo tương đồng **qWALS** (2026, dựa trên các đặc trưng của World Atlas of Language Structures) định lượng điều này. **LANGRANK** (Lin et al., ACL 2019) là một phương pháp riêng, ra đời sớm hơn, xếp hạng các ngôn ngữ nguồn ứng viên từ sự kết hợp của tương đồng ngôn ngữ, kích thước kho ngữ liệu và quan hệ phả hệ.

Quy tắc thực hành: nếu ngôn ngữ đích của bạn có một họ hàng gần về loại hình và nhiều tài nguyên, hãy thử fine-tune trên ngôn ngữ đó trước, rồi so sánh với fine-tune tiếng Anh.

```figure
n5-crosslingual-bridge
```

## Tự tay làm

### Bước 1: phân loại xuyên ngôn ngữ zero-shot

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Một mô hình, ba ngôn ngữ, cùng một API. XLM-R huấn luyện trên dữ liệu NLI chuyển giao tốt sang bài toán phân loại nhờ mánh entailment.

### Bước 2: không gian embedding đa ngôn ngữ

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Các bản dịch nằm gần nhau trong không gian embedding. Một câu tiếng Anh khác nghĩa thì nằm xa hơn. Đây là thứ khiến truy hồi, phân cụm và tính tương đồng xuyên ngôn ngữ hoạt động được.

### Bước 3: chiến lược fine-tune few-shot

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Với 100-500 ví dụ ngôn ngữ đích, `num_train_epochs=5` và `learning_rate=2e-5` là mặc định an toàn. Learning rate cao hơn làm sự căn chỉnh đa ngôn ngữ sụp đổ và bạn nhận về một mô hình chỉ biết tiếng Anh.

## Đánh giá thực sự hiệu quả

- **Độ chính xác theo từng ngôn ngữ trên tập giữ riêng.** Không gộp. Chỉ số gộp che mất phần đuôi dài.
- **Đối chiếu với mức cơ sở đơn ngữ.** Với ngôn ngữ có đủ dữ liệu, một mô hình đơn ngữ huấn luyện từ đầu đôi khi thắng mô hình đa ngôn ngữ. Hãy thử.
- **Kiểm tra ở mức thực thể.** Thực thể có tên trong ngôn ngữ đích. Mô hình đa ngôn ngữ thường tokenize yếu với những script (hệ chữ) xa tiếng Latinh.
- **Tính nhất quán xuyên ngôn ngữ.** Cùng một nghĩa ở hai ngôn ngữ phải cho cùng một dự đoán. Hãy đo khoảng cách.

## Áp dụng

Bộ công cụ của 2026:

| Tác vụ | Đề xuất |
|-----|-------------|
| Phân loại, 100 ngôn ngữ | XLM-R-base (~270M) đã fine-tune |
| Phân loại văn bản zero-shot | `joeddav/xlm-roberta-large-xnli` |
| Embedding câu đa ngôn ngữ | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Dịch, 200 ngôn ngữ | `facebook/nllb-200-distilled-600M` (xem bài 11) |
| Sinh đa ngôn ngữ | Claude, GPT-4, Aya-23, mT5-XXL |
| NLP cho ngôn ngữ ít tài nguyên | XLM-V hoặc fine-tune chuyên miền trên một ngôn ngữ nhiều tài nguyên có quan hệ |

Luôn dự trù ngân sách để fine-tune trong ngôn ngữ đích nếu hiệu năng quan trọng. Zero-shot là điểm khởi đầu, không phải câu trả lời cuối.

### Thuế tokenization (điều gì hỏng với ngôn ngữ ít tài nguyên)

Mô hình đa ngôn ngữ dùng chung một tokenizer cho tất cả ngôn ngữ của nó. Kho từ vựng đó được huấn luyện trên một kho ngữ liệu bị chi phối bởi tiếng Anh, Pháp, Tây Ban Nha, Trung, Đức. Với bất kỳ ngôn ngữ nào ngoài nhóm chi phối, ba loại thuế cộng dồn một cách âm thầm:

- **Thuế fertility (độ sinh token).** Văn bản ngôn ngữ ít tài nguyên bị tokenize thành nhiều token trên mỗi từ hơn hẳn tiếng Anh. Một câu tiếng Hindi có thể cần lượng token gấp 3-5 lần một câu tiếng Anh tương đương. Con số 3-5 lần đó ăn vào cửa sổ ngữ cảnh, hiệu quả huấn luyện và độ trễ của bạn.
- **Thuế khôi phục biến thể.** Mỗi lỗi gõ, biến thể dấu, sai lệch chuẩn hóa Unicode hay biến đổi chữ hoa chữ thường đều trở thành một chuỗi hoàn toàn không liên quan trong không gian embedding. Mô hình không thể học được những tương ứng chính tả mà người bản ngữ thấy là hiển nhiên.
- **Thuế tràn năng lực.** Thuế 1 và 2 tiêu thụ vị trí ngữ cảnh, độ sâu lớp và số chiều embedding. Phần còn lại cho suy luận thực sự vì thế nhỏ hơn một cách có hệ thống so với phần mà một ngôn ngữ nhiều tài nguyên nhận được từ cùng mô hình đó.

Triệu chứng thực tế: mô hình của bạn huấn luyện bình thường trên tiếng Hindi, đường cong loss trông ổn, perplexity khi đánh giá có vẻ hợp lý, và đầu ra trên production lại sai một cách tinh vi. Hình thái từ sụp giữa câu. Những biến tố hiếm không thể phục hồi. **Bạn không thể thoát khỏi một tokenizer hỏng bằng cách tăng dữ liệu.**

Cách giảm nhẹ: chọn tokenizer có độ phủ tốt cho ngôn ngữ đích (từ vựng 1M token của XLM-V là cách sửa trực tiếp); kiểm tra fertility của tokenization trên văn bản đích giữ riêng trước khi huấn luyện; dùng byte-level fallback (SentencePiece `byte_fallback=True`, BPE byte-level kiểu GPT-2) cho những hệ chữ đuôi dài thực sự để không gì bị OOV.

## Đóng gói

Lưu thành `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Bài tập

1. **Dễ.** Chạy pipeline phân loại zero-shot trên 10 câu mỗi ngôn ngữ với tiếng Anh, Pháp, Hindi và Ả Rập. Báo cáo độ chính xác trên từng ngôn ngữ. Bạn sẽ thấy tiếng Pháp mạnh, tiếng Hindi khá, tiếng Ả Rập dao động.
2. **Trung bình.** Dùng `paraphrase-multilingual-MiniLM-L12-v2` để dựng một bộ truy hồi xuyên ngôn ngữ trên một kho ngữ liệu nhỏ trộn nhiều ngôn ngữ. Truy vấn bằng tiếng Anh, truy hồi tài liệu ở bất kỳ ngôn ngữ nào. Đo recall@5.
3. **Khó.** So sánh fine-tune với nguồn tiếng Anh và nguồn tiếng Hindi cho một tác vụ phân loại tiếng Hindi. Dùng 500 ví dụ ngôn ngữ đích để fine-tune few-shot trong cả hai chế độ. Báo cáo nguồn nào cho độ chính xác tiếng Hindi tốt hơn và hơn bao nhiêu. Đây chính là luận điểm LANGRANK thu nhỏ.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Multilingual model | Một mô hình, nhiều ngôn ngữ | Từ vựng và tham số dùng chung trên các ngôn ngữ. |
| Cross-lingual transfer | Huấn luyện trên một ngôn ngữ, chạy trên ngôn ngữ khác | Fine-tune trên ngôn ngữ nguồn, đánh giá trên ngôn ngữ đích mà không có nhãn ngôn ngữ đích. |
| Zero-shot | Không có nhãn ngôn ngữ đích | Chuyển giao mà không fine-tune trên ngôn ngữ đích. |
| Few-shot | Ít nhãn ngôn ngữ đích | 100-500 ví dụ ngôn ngữ đích dùng để fine-tune. |
| mBERT | LM đa ngôn ngữ đầu tiên | BERT 104 ngôn ngữ tiền huấn luyện trên Wikipedia. |
| XLM-R | Mức cơ sở xuyên ngôn ngữ chuẩn | RoBERTa 100 ngôn ngữ tiền huấn luyện trên CommonCrawl. |
| NLLB | MT 200 ngôn ngữ của Meta | No Language Left Behind. Gồm 55 ngôn ngữ ít tài nguyên. |

## Đọc thêm

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — bài báo XLM-R.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — bài phân tích đã khởi đầu hướng nghiên cứu chuyển giao xuyên ngôn ngữ.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — bài báo NLLB-200.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya, LLM đa ngôn ngữ của Cohere.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — bài báo về ngôn ngữ nguồn qWALS / LANGRANK.

---

[← Bài trước: Chatbot](../../17-chatbots-rule-to-neural/docs/vi.md) · [Bài sau: Token hoá dưới từ →](../../19-subword-tokenization/docs/vi.md)
