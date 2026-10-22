# BERT — Masked language modeling (mô hình ngôn ngữ có mặt nạ)

> GPT dự đoán từ tiếp theo. BERT dự đoán một từ bị thiếu. Chỉ một câu khác biệt — và nửa thập kỷ của mọi thứ mang hình dạng embedding.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 5, Bài 02 (Biểu diễn văn bản)
**Time:** ~45 phút

## Vấn đề

Năm 2018, mọi bài toán NLP — sentiment, NER, QA, entailment — đều tự huấn luyện mô hình riêng từ đầu trên dữ liệu gán nhãn của chính nó. Không có checkpoint "hiểu tiếng Anh" nào đã tiền huấn luyện để bạn tinh chỉnh (fine-tuning). ELMo (2018) cho thấy có thể tiền huấn luyện embedding ngữ cảnh bằng LSTM bidirectional (hai chiều); nó giúp ích nhưng không khái quát hoá được.

BERT (Devlin và cộng sự 2018) đặt câu hỏi: nếu ta lấy một transformer encoder, huấn luyện nó trên mọi câu trên internet, và buộc nó dự đoán những từ bị thiếu từ ngữ cảnh ở cả hai phía thì sao? Sau đó bạn tinh chỉnh một head cho tác vụ hạ nguồn của mình. Hiệu quả tham số là một phát hiện chấn động.

Kết quả: trong vòng 18 tháng, BERT và các biến thể của nó (RoBERTa, ALBERT, ELECTRA) thống trị mọi bảng xếp hạng NLP đang tồn tại. Đến 2020, mọi công cụ tìm kiếm, pipeline kiểm duyệt nội dung và hệ thống tìm kiếm ngữ nghĩa trên trái đất đều có một BERT bên trong.

Năm 2026, các mô hình encoder-only (chỉ encoder) vẫn là công cụ đúng cho phân loại, truy xuất và trích xuất có cấu trúc — chúng chạy nhanh hơn 5–10× mỗi token so với decoder, và embedding của chúng là xương sống của mọi ngăn xếp truy xuất hiện đại. ModernBERT (tháng 12 năm 2024) đẩy kiến trúc lên ngữ cảnh 8K với Flash Attention + RoPE + GeGLU.

## Khái niệm

![Mô hình ngôn ngữ có mặt nạ: chọn token, che chúng, dự đoán bản gốc](../assets/bert-mlm.svg)

### Tín hiệu huấn luyện

Lấy một câu: `the quick brown fox jumps over the lazy dog`.

Che ngẫu nhiên 15% số token:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Huấn luyện mô hình dự đoán các token gốc tại những vị trí bị che. Vì encoder là bidirectional, việc dự đoán `[MASK]` ở vị trí 1 có thể dùng `brown fox jumps` ở các vị trí 2 trở đi. Đó chính là điều GPT không làm được.

### Quy tắc mặt nạ của BERT

Trong 15% số token được chọn để dự đoán:

- 80% bị thay bằng `[MASK]`.
- 10% bị thay bằng một token ngẫu nhiên.
- 10% được giữ nguyên.

Vì sao không luôn dùng `[MASK]`? Vì `[MASK]` không bao giờ xuất hiện lúc suy luận. Huấn luyện mô hình quen với việc `[MASK]` có mặt ở 100% vị trí bị che sẽ tạo ra dịch chuyển phân phối giữa tiền huấn luyện và tinh chỉnh. 10% ngẫu nhiên + 10% giữ nguyên giúp mô hình trung thực.

### Next sentence prediction (dự đoán câu kế) — và vì sao nó bị bỏ

BERT gốc còn huấn luyện NSP: cho hai câu A và B, dự đoán xem B có nối tiếp A không. RoBERTa (2019) loại bỏ nó và cho thấy NSP làm hại chứ không giúp gì. Các encoder hiện đại bỏ qua nó.

### Điều gì đã thay đổi vào năm 2026: ModernBERT

Bài báo ModernBERT 2024 xây lại block với các primitive của 2026:

| Thành phần | BERT gốc (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Vị trí | Tuyệt đối học được | RoPE |
| Kích hoạt | GELU | GeGLU |
| Chuẩn hoá | LayerNorm | Pre-norm RMSNorm |
| Attention | Dense đầy đủ | Xen kẽ local (128) + global |
| Độ dài ngữ cảnh | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

Và khác với ngăn xếp 2018, nó là Flash-Attention-native. Suy luận nhanh hơn 2–3× ở độ dài chuỗi 8K so với DeBERTa-v3, với điểm GLUE tốt hơn.

### Những trường hợp vẫn chọn encoder vào năm 2026

| Tác vụ | Vì sao encoder thắng decoder |
|------|---------------------------|
| Truy xuất / embedding tìm kiếm ngữ nghĩa | Ngữ cảnh bidirectional = chất lượng embedding tốt hơn trên mỗi token |
| Phân loại (sentiment, intent, độc hại) | Một lượt forward; không có chi phí sinh token |
| NER / gán nhãn token | Đầu ra theo từng vị trí, bidirectional từ gốc |
| Suy luận zero-shot (NLI) | Head phân loại đặt trên đỉnh encoder |
| Reranker cho RAG | Chấm điểm cross-encoder, nhanh hơn 10x so với LLM reranker |

```figure
transformer-residual
```

## Tự tay làm

### Bước 1: logic tạo mặt nạ

Xem `code/main.py`. Hàm `create_mlm_batch` nhận một danh sách token ID, kích thước vocab và xác suất che. Trả về input ID (đã áp mặt nạ) và label (chỉ ở các vị trí bị che, -100 ở nơi khác — quy ước ignore index của PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Bước 2: chạy dự đoán MLM trên một kho ngữ liệu tí hon

Huấn luyện một encoder 2 lớp + MLM head trên vocab 20 từ, 200 câu. Không dùng gradient — ta chỉ kiểm tra hợp lý bằng forward pass. Huấn luyện đầy đủ cần PyTorch.

### Bước 3: so sánh các kiểu mặt nạ

Cho thấy quy tắc ba nhánh giữ mô hình dùng được mà không cần `[MASK]`. Dự đoán trên một câu không bị che và một câu bị che. Cả hai đều nên cho phân phối token hợp lý vì mô hình đã thấy cả hai kiểu mẫu khi huấn luyện.

### Bước 4: tinh chỉnh head

Thay MLM head bằng một head phân loại trên bộ dữ liệu sentiment đồ chơi. Chỉ head được huấn luyện; encoder bị đóng băng. Đây là khuôn mẫu mà mọi ứng dụng BERT theo.

## Áp dụng

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Mô hình embedding là BERT đã tinh chỉnh.** Các mô hình `sentence-transformers` như `all-MiniLM-L6-v2` là BERT được huấn luyện với contrastive loss. Encoder vẫn thế. Chỉ loss thay đổi.

**Cross-encoder reranker cũng là BERT đã tinh chỉnh.** Phân loại cặp trên `[CLS] query [SEP] doc [SEP]`. Attention bidirectional giữa query và doc chính là thứ mang lại cho cross-encoder lợi thế chất lượng so với biencoder.

**Khi nào không nên chọn BERT vào năm 2026.** Bất cứ thứ gì mang tính sinh. Encoder không có cách nào hợp lý để sinh token tự hồi quy. Và cả: bất cứ thứ gì dưới 1B tham số khi một decoder nhỏ có thể đạt chất lượng tương đương mà linh hoạt hơn (Phi-3-Mini, Qwen2-1.5B).

## Đóng gói

Xem `outputs/skill-bert-finetuner.md`. Skill này phạm vi hoá một lần tinh chỉnh BERT (chọn backbone, đặc tả head, dữ liệu, eval, điểm dừng) cho một bài toán phân loại hoặc trích xuất mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py` và in phân phối mặt nạ trên 10.000 token. Xác nhận ~15% được chọn, và trong số đó ~80% trở thành `[MASK]`.
2. **Trung bình.** Cài đặt whole-word masking: nếu một từ được tokenize thành nhiều subword, che tất cả subword cùng nhau hoặc không che subword nào. Đo xem điều này có cải thiện độ chính xác MLM trên kho 500 câu không.
3. **Khó.** Huấn luyện một BERT tí hon (2 lớp, d=64) trên 10.000 câu từ một bộ dữ liệu công khai. Tinh chỉnh token `[CLS]` cho sentiment SST-2. So sánh với baseline chỉ decoder có số tham số khớp — bên nào thắng?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | Tín hiệu huấn luyện: thay ngẫu nhiên 15% token bằng `[MASK]`, dự đoán bản gốc. |
| Bidirectional | "Nhìn cả hai phía" | Attention của encoder không có causal mask — mỗi vị trí thấy mọi vị trí khác. |
| `[CLS]` | "Token pooler" | Token đặc biệt gắn ở đầu mỗi chuỗi; embedding cuối của nó được dùng làm biểu diễn mức câu. |
| `[SEP]` | "Dấu phân tách đoạn" | Tách các chuỗi ghép cặp (ví dụ query/doc, câu A/B). |
| NSP | "Next sentence prediction" | Nhiệm vụ tiền huấn luyện thứ hai của BERT; RoBERTa cho thấy nó vô dụng, bị bỏ sau 2019. |
| Fine-tuning | "Thích nghi với một tác vụ" | Giữ encoder gần như đóng băng; huấn luyện một head nhỏ trên đỉnh cho tác vụ hạ nguồn. |
| Cross-encoder | "Một reranker" | Một BERT nhận cả query và doc làm đầu vào, xuất ra điểm liên quan. |
| ModernBERT | "Bản làm mới 2024" | Encoder xây lại với RoPE, RMSNorm, GeGLU, attention local/global xen kẽ, ngữ cảnh 8K. |

## Đọc thêm

- [Devlin và cộng sự (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — bài báo gốc.
- [Liu và cộng sự (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — cách huấn luyện BERT cho đúng; loại bỏ NSP.
- [Clark và cộng sự (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — phát hiện token bị thay thế đánh bại MLM với cùng lượng compute.
- [Warner và cộng sự (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — bài báo ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — tham chiếu encoder chuẩn mực.

---

[← Bài trước: Transformer đầy đủ](../../05-full-transformer/docs/vi.md) · [Bài sau: GPT →](../../07-gpt-causal-language-modeling/docs/vi.md)
