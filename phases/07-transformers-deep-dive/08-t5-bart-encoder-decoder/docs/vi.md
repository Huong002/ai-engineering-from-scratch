# T5, BART — Mô hình encoder-decoder

> Encoder hiểu. Decoder sinh. Ghép chúng lại với nhau và bạn có một mô hình dựng cho các tác vụ đầu vào → đầu ra: dịch, tóm tắt, viết lại, phiên âm.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 7, Bài 06 (BERT), Giai đoạn 7, Bài 07 (GPT)
**Time:** ~45 phút

## Vấn đề

GPT chỉ có decoder và BERT chỉ có encoder, mỗi mô hình lược bớt kiến trúc năm 2017 theo một mục tiêu khác nhau. Nhưng nhiều tác vụ vốn dĩ là đầu vào-đầu ra:

- Dịch: tiếng Anh → tiếng Pháp.
- Tóm tắt: bài viết 5.000 token → bản tóm tắt 200 token.
- Nhận dạng giọng nói: token âm thanh → token văn bản.
- Trích xuất có cấu trúc: văn xuôi → JSON.

Với những tác vụ này, encoder-decoder (mã hoá-giải mã) là lựa chọn khớp nhất. Encoder tạo ra một biểu diễn dày đặc của nguồn. Decoder sinh đầu ra, cross-attend vào biểu diễn đó ở mọi bước. Huấn luyện là dịch một vị trí ở phía đầu ra. Cùng loss như GPT, chỉ khác là được điều kiện hoá trên đầu ra của encoder.

Hai bài báo đã định hình cách làm hiện đại:

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." Mọi tác vụ NLP được đóng khung lại thành text-to-text (văn bản sang văn bản). Một kiến trúc, một từ vựng, một loss. Tiền huấn luyện bằng dự đoán span bị che (làm hỏng các span ở đầu vào, giải mã chúng ở đầu ra).
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." Autoencoder khử nhiễu (denoising): làm hỏng đầu vào theo nhiều cách (xáo trộn, che, xoá, xoay), rồi yêu cầu decoder tái tạo bản gốc.

Năm 2026, định dạng encoder-decoder vẫn sống ở những nơi cấu trúc đầu vào quan trọng:

- Whisper (giọng nói → văn bản).
- Stack dịch của Google.
- Một số mô hình hoàn thiện / sửa lỗi code có cấu trúc ngữ cảnh-và-chỉnh sửa riêng.
- Flan-T5 và các biến thể cho tác vụ suy luận có cấu trúc.

Decoder-only đã chiếm sân khấu chính, nhưng encoder-decoder chưa bao giờ biến mất.

## Khái niệm

![Mã hoá-giải mã với cross-attention](../assets/encoder-decoder.svg)

### Vòng lan truyền xuôi

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Điểm then chốt: encoder chạy một lần cho mỗi đầu vào. Decoder chạy tự hồi quy nhưng cross-attend vào *cùng* đầu ra của encoder ở mọi bước. Cache đầu ra của encoder là một cách tăng tốc miễn phí với đầu vào dài.

### Tiền huấn luyện T5 — span corruption

Chọn ngẫu nhiên các span của đầu vào (độ dài trung bình 3 token, tổng cộng 15%). Thay mỗi span bằng một sentinel duy nhất: `<extra_id_0>`, `<extra_id_1>`, v.v. Decoder chỉ sinh các span bị làm hỏng (span corruption) kèm sentinel làm tiền tố (prefix) của chúng:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Tín hiệu rẻ hơn so với dự đoán toàn bộ chuỗi. Cạnh tranh với MLM (BERT) và prefix-LM (UniLM) trong phần ablation của bài báo T5.

### Tiền huấn luyện BART — khử nhiễu đa dạng nhiễu

BART thử năm hàm tạo nhiễu:

1. Che token.
2. Xoá token.
3. Điền vào chỗ trống (che một span, decoder chèn đúng độ dài).
4. Hoán vị câu.
5. Xoay tài liệu.

Kết hợp điền chỗ trống + hoán vị câu cho kết quả hạ nguồn tốt nhất. Decoder luôn tái tạo bản gốc. Đầu ra của BART là toàn bộ chuỗi, không chỉ các span bị làm hỏng — nên chi phí tính toán tiền huấn luyện cao hơn T5.

### Suy luận

Sinh tự hồi quy giống GPT. Greedy / beam / top-p sampling đều áp dụng được. Beam search (độ rộng 4–5) là chuẩn cho dịch và tóm tắt vì phân phối đầu ra hẹp hơn chat.

### Khi nào chọn biến thể nào trong năm 2026

| Tác vụ | Encoder-decoder? | Vì sao |
|------|------------------|-----|
| Dịch | Có, thường vậy | Chuỗi nguồn rõ ràng; phân phối đầu ra cố định; beam search hiệu quả |
| Giọng nói sang văn bản | Có (Whisper) | Modality đầu vào khác đầu ra; encoder định hình đặc trưng âm thanh |
| Chat / suy luận | Không, decoder-only | Không có "đầu vào" cố định — hội thoại chính là chuỗi |
| Hoàn thiện code | Thường không | Decoder-only với ngữ cảnh dài thắng; các mô hình code như Qwen 2.5 Coder là decoder-only |
| Tóm tắt | Cả hai đều được | BART, PEGASUS vượt các baseline decoder-only trước đó; LLM decoder-only hiện đại bắt kịp chúng |
| Trích xuất có cấu trúc | Cả hai | T5 gọn vì "text → text" nuốt được mọi định dạng đầu ra |

Xu hướng từ khoảng 2022: decoder-only tiếp quản những tác vụ mà trước đây encoder-decoder làm chủ vì (a) LLM decoder-only tinh chỉnh theo chỉ dẫn tổng quát hoá được mọi thứ qua prompt, (b) một kiến trúc dễ mở rộng hơn hai kiến trúc, (c) RLHF giả định có decoder. Encoder-decoder trụ lại ở nơi modality đầu vào khác biệt (giọng nói, hình ảnh) hoặc nơi chất lượng beam search quan trọng.

```figure
encoder-decoder
```

## Tự tay làm

Xem `code/main.py`. Chúng ta cài đặt span corruption kiểu T5 cho một corpus đồ chơi — phần hữu ích nhất của bài này vì nó xuất hiện trong mọi công thức tiền huấn luyện encoder-decoder kể từ đó.

### Bước 1: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Định dạng target theo quy ước T5: `<sent0> span0 <sent1> span1 ...`. Đầu vào bị làm hỏng xen kẽ các token không đổi với các token sentinel tại vị trí của span.

### Bước 2: kiểm tra khứ hồi

Cho đầu vào bị làm hỏng và target, hãy tái tạo câu gốc. Nếu phép làm hỏng của bạn đảo ngược được thì lượt lan truyền xuôi là xác định rõ. Đây là một kiểm tra hợp lý — huấn luyện thật không bao giờ làm việc này, nhưng test rẻ và bắt được lỗi lệch một đơn vị trong khâu đếm span.

### Bước 3: tạo nhiễu BART

Năm hàm: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Ghép hai hàm bất kỳ và cho xem kết quả.

## Áp dụng

Tham chiếu HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Mẹo của T5: tên tác vụ đi vào văn bản đầu vào. Cùng một mô hình xử lý hàng chục tác vụ vì mỗi tác vụ là văn bản vào, văn bản ra. Năm 2026 mẫu hình này đã được tổng quát hoá bởi các mô hình decoder-only tinh chỉnh theo chỉ dẫn, nhưng T5 đã hệ thống hoá nó trước.

## Đóng gói

Xem `outputs/skill-seq2seq-picker.md`. Skill chọn giữa encoder-decoder và decoder-only cho một tác vụ mới dựa trên cấu trúc đầu vào-đầu ra, độ trễ và mục tiêu chất lượng.

## Bài tập

1. **Dễ.** Chạy `code/main.py`, áp span corruption lên một câu 30 token, kiểm tra rằng nối các token nguồn không phải sentinel với các span target đã giải mã sẽ tái tạo câu gốc.
2. **Trung bình.** Cài đặt nhiễu `text_infill` của BART: thay các span ngẫu nhiên bằng một token `<mask>` duy nhất, và decoder phải suy ra đúng độ dài span cùng nội dung. Cho một ví dụ.
3. **Khó.** Tinh chỉnh `flan-t5-small` trên một corpus tiếng Anh → pig-Latin nhỏ (200 cặp). Đo BLEU trên tập giữ lại 50 cặp. So sánh với việc tinh chỉnh `Llama-3.2-1B` trên cùng dữ liệu với cùng compute.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq transformer" | Hai stack: encoder hai chiều cho đầu vào, decoder nhân quả với cross-attention cho đầu ra. |
| Cross-attention | "Nơi nguồn nói chuyện với đích" | Q của decoder × K/V của encoder. Nơi duy nhất thông tin encoder đi vào decoder. |
| Span corruption | "Chiêu tiền huấn luyện của T5" | Thay các span ngẫu nhiên bằng token sentinel; decoder sinh ra các span đó. |
| Denoising objective | "Trò chơi của BART" | Áp một hàm nhiễu lên đầu vào, huấn luyện decoder tái tạo chuỗi sạch. |
| Sentinel token | "Chỗ giữ chỗ `<extra_id_N>`" | Token đặc biệt đánh dấu các span bị làm hỏng ở nguồn và đánh dấu lại ở target. |
| Flan | "T5 tinh chỉnh theo chỉ dẫn" | T5 tinh chỉnh trên hơn 1.800 tác vụ; làm encoder-decoder cạnh tranh được ở việc theo chỉ dẫn. |
| Beam search | "Chiến lược giải mã" | Giữ top-k chuỗi bộ phận ở mỗi bước; chuẩn cho dịch/tóm tắt. |
| Teacher forcing | "Đầu vào lúc huấn luyện (teacher forcing)" | Khi huấn luyện, đưa token đầu ra thật của bước trước vào decoder, không phải token lấy mẫu. |

## Đọc thêm

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, mô hình encoder-decoder chuẩn mực của năm 2026.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — bản cài đặt tham chiếu.

---

[← Bài trước: GPT](../../07-gpt-causal-language-modeling/docs/vi.md) · [Bài sau: Vision Transformers (ViT) →](../../09-vision-transformers/docs/vi.md)
