# Dịch máy

> Dịch là bài toán đã nuôi sống nghiên cứu NLP suốt ba mươi năm và vẫn đang nuôi sống nó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 10 (Cơ chế attention), Giai đoạn 5, Bài 04 (GloVe, FastText, Subword)
**Time:** ~75 phút

## Vấn đề

Một mô hình đọc một câu ở ngôn ngữ này và tạo ra một câu ở ngôn ngữ khác. Độ dài thay đổi. Trật tự từ thay đổi. Có từ ở bản nguồn ánh xạ sang nhiều từ ở bản đích và ngược lại. Thành ngữ không chịu ánh xạ một-một. "I miss you" trong tiếng Pháp là "tu me manques" — nghĩa đen là "bạn đang thiếu đối với tôi". Không có căn chỉnh ở mức từ nào sống sót qua chuyện đó.

machine translation (dịch máy) là bài toán đã buộc NLP phải phát minh ra encoder-decoder, attention, transformer, và cuối cùng là cả hệ hình LLM. Mỗi bước tiến đều đến vì chất lượng dịch đo được và khoảng cách giữa người và máy rất cứng đầu.

Bài này bỏ qua phần lịch sử và dạy pipeline đang chạy của năm 2026: encoder-decoder đa ngữ đã tiền huấn luyện (NLLB-200 hoặc mBART), tokenize subword, beam search, đánh giá BLEU và chrF, cùng vài kiểu hỏng vẫn lọt ra production mà không ai bắt được.

## Khái niệm

![Pipeline MT: tokenize → encode → decode với attention → detokenize](../assets/mt-pipeline.svg)

MT hiện đại là một transformer encoder-decoder huấn luyện trên parallel corpus (ngữ liệu song song). Encoder đọc bản nguồn theo cách tokenize của ngôn ngữ đó. Decoder sinh bản đích, từng subword một, dùng đầu ra của encoder qua cross-attention (bài 10). Quá trình decoding (giải mã) dùng beam search để tránh bẫy greedy decoding. Đầu ra trải qua detokenization (ghép token), khôi phục chữ hoa, rồi được chấm điểm so với bản tham chiếu.

Ba lựa chọn vận hành quyết định chất lượng MT trong thực tế.

- **Tokenizer.** SentencePiece BPE huấn luyện trên corpus trộn nhiều ngôn ngữ. Từ vựng dùng chung giữa các ngôn ngữ chính là thứ cho phép các cặp zero-shot trong NLLB.
- **Kích thước mô hình.** NLLB-200 distilled 600M chạy vừa trên laptop. NLLB-200 3.3B là mức mặc định production đã công bố. 54.5B là trần của nghiên cứu.
- **Decoding.** Beam width 4-5 cho nội dung chung. Length penalty để tránh đầu ra quá ngắn. Constrained decoding khi bạn cần nhất quán thuật ngữ.

```figure
seq2seq-alignment
```

## Tự tay làm

### Bước 1: một lệnh gọi MT đã tiền huấn luyện

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Ba điều quan trọng ở đây. `src_lang` nói cho tokenizer biết dùng hệ chữ và cách tách từ nào. `forced_bos_token_id` nói cho decoder biết phải sinh ra ngôn ngữ nào. Cả hai đều là chiêu riêng của NLLB; mBART và M2M-100 dùng quy ước riêng của chúng và không hoán đổi được cho nhau.

### Bước 2: BLEU và chrF

BLEU đo mức trùng n-gram giữa đầu ra (hypothesis — giả thuyết dịch) và bản tham chiếu. Bốn kích thước n-gram tham chiếu (1-4), trung bình nhân của các precision, brevity penalty cho đầu ra quá ngắn. Điểm nằm trong [0, 100]. Được dùng rộng rãi. Khó diễn giải: 30 BLEU là "dùng được"; 40 là "tốt"; 50 là "xuất sắc"; chênh lệch dưới 1 BLEU là nhiễu.

chrF đo F-score ở mức ký tự. Nhạy hơn với các ngôn ngữ giàu hình thái, nơi BLEU đếm thiếu các cặp khớp. Thường được báo cáo cùng với BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Luôn dùng `sacrebleu`. Nó chuẩn hoá cách tokenize để điểm số so sánh được giữa các bài báo. Tự viết lại phép tính BLEU là cách tạo ra những benchmark gây nhầm lẫn.

### Hệ thống đánh giá ba tầng (2026)

Đánh giá MT hiện đại dùng ba họ metric bổ trợ cho nhau. Hãy ship kèm ít nhất hai.

- **Heuristic** (BLEU, chrF). Nhanh, dựa trên bản tham chiếu, dễ hiểu, không nhạy với việc diễn đạt lại. Dùng để so sánh với kết quả cũ và phát hiện hồi quy.
- **Học được** (COMET, BLEURT, BERTScore). Các mô hình neural huấn luyện trên đánh giá của con người; chúng so sánh độ tương đồng ngữ nghĩa giữa bản dịch với bản nguồn và bản tham chiếu. COMET có tương quan cao nhất với nghiên cứu MT từ 2023 và là mặc định production năm 2026 ở những nơi chất lượng là quan trọng.
- **LLM làm giám khảo** (không cần bản tham chiếu). Prompt một mô hình lớn để chấm bản dịch theo fluency (độ trôi chảy), adequacy (độ trung thành), giọng điệu, mức phù hợp văn hoá. GPT-4 làm giám khảo khớp với đánh giá của con người khoảng ~80% số lần khi rubric được thiết kế tốt. Dùng cho nội dung mở, nơi không tồn tại bản tham chiếu.

Stack thực dụng năm 2026: `sacrebleu` cho BLEU và chrF, `unbabel-comet` cho COMET, và một LLM được prompt để cho tín hiệu cuối cùng hướng tới con người. Hãy hiệu chỉnh mọi metric trên 50-100 ví dụ có nhãn của con người trước khi tin nó trên dữ liệu production.

Các metric không cần bản tham chiếu (COMET-QE, BLEURT-QE, LLM làm giám khảo) cho phép bạn đánh giá bản dịch mà không có bản tham chiếu, điều này quan trọng với các cặp ngôn ngữ đuôi dài, nơi không tồn tại bản dịch tham chiếu.

### Bước 3: những gì hỏng trong production

Pipeline ở trên sẽ dịch trôi chảy trong 80% số lần và hỏng âm thầm ở 20% còn lại. Các kiểu hỏng đã được đặt tên:

- **Hallucination.** Mô hình bịa ra nội dung không có trong bản nguồn. Phổ biến với từ vựng chuyên ngành lạ. Triệu chứng: đầu ra trôi chảy nhưng khẳng định những điều bản nguồn không hề nói. Cách giảm: constrained decoding trên thuật ngữ chuyên ngành, người duyệt với nội dung bị quản lý, giám sát đầu ra dài hơn đầu vào nhiều.
- **Sinh lệch đích.** Mô hình dịch sang nhầm ngôn ngữ. NLLB khá dễ mắc lỗi này với các cặp ngôn ngữ hiếm. Cách giảm: kiểm tra `forced_bos_token_id` và luôn giải mã kèm bước kiểm tra nhận dạng ngôn ngữ trên đầu ra.
- **Trôi thuật ngữ.** "Sign up" thành "s'inscrire" ở tài liệu 1 và "créer un compte" ở tài liệu 2. Với chữ trên UI và các chuỗi hiển thị cho người dùng, nhất quán quan trọng hơn chất lượng thô. Cách giảm: constrained decoding theo glossary hoặc từ điển hậu biên tập.
- **Lệch mức trang trọng.** "tu" so với "vous" trong tiếng Pháp, các mức lịch sự trong tiếng Nhật. Mô hình chọn dạng nào phổ biến hơn trong dữ liệu huấn luyện. Với nội dung hướng tới khách hàng, điều này thường là sai. Cách giảm: thêm tiền tố prompt bằng một token mức trang trọng nếu mô hình hỗ trợ, hoặc fine-tune một mô hình nhỏ trên corpus chỉ toàn văn trang trọng.
- **Độ dài bùng nổ với đầu vào ngắn.** Câu nguồn rất ngắn thường sinh ra bản dịch quá dài vì length penalty rơi thẳng đứng dưới khoảng 5 token nguồn. Cách giảm: đặt trần độ dài cứng tỉ lệ với độ dài bản nguồn.

### Bước 4: fine-tune cho một lĩnh vực

Mô hình đã tiền huấn luyện là dân tổng quát. Dịch pháp lý, y tế, hay hội thoại trong game đều được lợi rõ rệt khi fine-tune trên dữ liệu song song chuyên ngành. Công thức không có gì lạ:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Vài nghìn ví dụ song song chất lượng cao thắng vài trăm nghìn ví dụ nhiễu cào từ web. Chất lượng dữ liệu huấn luyện là đòn bẩy production lớn nhất.

## Áp dụng

Stack production cho MT năm 2026:

| Trường hợp sử dụng | Điểm bắt đầu nên chọn |
|---------|---------------------------|
| Mọi ngôn ngữ sang mọi ngôn ngữ, 200 ngôn ngữ | `facebook/nllb-200-distilled-600M` (laptop) hoặc `nllb-200-3.3B` (production) |
| Lấy tiếng Anh làm trung tâm, chất lượng cao, 50 ngôn ngữ | `facebook/mbart-large-50-many-to-many-mmt` |
| Chạy ngắn, suy luận rẻ, Anh-Pháp/Đức/Tây Ban Nha | Các mô hình Helsinki-NLP / Marian |
| Then chốt về độ trễ, chạy phía trình duyệt | Marian lượng tử hoá ONNX (~50 MB) |
| Chất lượng tối đa, sẵn sàng trả tiền | GPT-4 / Claude / Gemini với prompt dịch |

Tính đến 2026, LLM đã vượt các mô hình MT chuyên dụng trên vài cặp ngôn ngữ, đặc biệt với nội dung giàu thành ngữ và ngữ cảnh dài. Cái giá phải trả là chi phí theo token và độ trễ. Hãy chọn LLM khi độ dài ngữ cảnh, tính nhất quán phong cách, hoặc thích ứng lĩnh vực qua prompt quan trọng hơn thông lượng.

## Đóng gói

Lưu thành `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Bài tập

1. **Dễ.** Dịch một đoạn tiếng Anh 5 câu sang tiếng Pháp rồi dịch ngược lại tiếng Anh bằng `nllb-200-distilled-600M`. Đo xem vòng khứ hồi gần với bản gốc đến mức nào. Bạn sẽ thấy ngữ nghĩa được giữ nguyên nhưng cách chọn từ bị trôi.
2. **Trung bình.** Cài một bước kiểm tra nhận dạng ngôn ngữ trên đầu ra bản dịch bằng `fasttext lid.176` hoặc `langdetect`. Tích hợp vào lệnh gọi MT để các trường hợp sinh lệch đích bị bắt trước khi trả về.
3. **Khó.** Fine-tune `nllb-200-distilled-600M` trên một corpus song song 5.000 cặp do bạn chọn. Đo BLEU trên tập giữ riêng trước và sau khi fine-tune. Báo cáo loại câu nào cải thiện và loại nào tệ đi.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| BLEU | Điểm chất lượng dịch | Precision n-gram kèm brevity penalty. [0, 100]. |
| chrF | F-score ở mức ký tự | F-score ở mức ký tự. Nhạy hơn với các ngôn ngữ giàu hình thái. |
| NMT | Dịch máy neural | Transformer encoder-decoder huấn luyện trên parallel corpus. Mặc định từ 2017 trở đi. |
| NLLB | No Language Left Behind | Họ mô hình MT 200 ngôn ngữ của Meta. |
| Constrained decoding | Đầu ra bị điều khiển | Ép một số token hoặc n-gram cụ thể xuất hiện / không xuất hiện trong đầu ra. |
| Hallucination | Nội dung bịa ra | Đầu ra của mô hình không được bản nguồn hỗ trợ. |

## Đọc thêm

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — bài báo NLLB.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — vì sao `sacrebleu` là cách duy nhất đúng để báo cáo BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — bài báo chrF.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — hướng dẫn fine-tune thực hành.

---

[← Bài trước: Cơ chế chú ý (attention)](../../10-attention-mechanism/docs/vi.md) · [Bài sau: Tóm tắt văn bản →](../../12-text-summarization/docs/vi.md)
