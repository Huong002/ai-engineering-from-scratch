# Whisper — Kiến trúc & Tinh chỉnh

> Whisper là một Transformer encoder (bộ mã hoá) — decoder (bộ giải mã) với cửa sổ 30 giây, được huấn luyện trên 680k giờ cặp audio-text đa ngôn ngữ gán nhãn yếu. Một kiến trúc, nhiều tác vụ, bền vững trên 99 ngôn ngữ. Chuẩn tham chiếu ASR của năm 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 04 (ASR), Giai đoạn 5, Bài 10 (Cơ chế attention), Giai đoạn 7, Bài 05 (Transformer đầy đủ)
**Time:** ~75 phút

## Vấn đề

Whisper, do OpenAI phát hành tháng 9 năm 2022, là mô hình ASR đầu tiên được đóng gói như một món hàng phổ thông: dán audio vào, nhận text ra, 99 ngôn ngữ, bền với tiếng ồn, chạy được trên laptop. Đến 2024, OpenAI đã ra mắt các biến thể Large-v3 và Turbo; đến 2026, Whisper là baseline mặc định cho mọi thứ từ chép podcast đến trợ lý giọng nói đến phụ đề YouTube.

Nhưng Whisper không phải một pipeline mà bạn có thể coi là hộp đen mãi mãi. Domain shift giết nó — thuật ngữ kỹ thuật, giọng vùng miền, danh từ riêng, clip ngắn, khoảng lặng. Bạn cần biết:

1. Bên trong nó thực chất là gì.
2. Cách đưa cho nó audio chia đoạn, streaming, hoặc audio dài một cách đúng đắn.
3. Khi nào nên fine-tune (tinh chỉnh) và làm thế nào.

## Khái niệm

![Kiến trúc Whisper encoder-decoder, các tác vụ, suy luận theo đoạn, tinh chỉnh](../assets/whisper.svg)

**Kiến trúc.** Transformer encoder-decoder chuẩn.

- Đầu vào: phổ log-mel 30 giây, 80 mel, hop 10 ms → 3000 frame. Clip ngắn hơn được đệm số 0, clip dài hơn được chia đoạn.
- Encoder: conv-downsample (stride 2) + `N` khối transformer. Với Large-v3: 32 lớp, 1280 chiều, 20 head.
- Decoder: `N` khối transformer với causal self-attn + cross-attn tới đầu ra của encoder. Kích thước bằng encoder.
- Đầu ra: token BPE trên vocab 51.865 token.

Large-v3 có 1,55B tham số. Turbo dùng decoder 4 lớp (thay vì 32), giảm độ trễ 8× với mức giảm chất lượng dưới 1% WER.

**Định dạng prompt.** Whisper là mô hình multitask (đa tác vụ) được điều khiển bằng các token đặc biệt trong prompt của decoder:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — thẻ ngôn ngữ; quyết định hành vi dịch hay chép chính tả.
- `<|transcribe|>` hoặc `<|translate|>` — dịch đầu ra tiếng Anh từ đầu vào bất kỳ ngôn ngữ nào, hoặc chép nguyên văn.
- `<|notimestamps|>` — bỏ qua timestamp (mốc thời gian) ở mức từ (nhanh hơn).

Prompt chính là thứ cho phép một mô hình làm nhiều tác vụ. Đổi `<|en|>` thành `<|fr|>` là nó chép tiếng Pháp.

**Cửa sổ 30 giây.** Mọi thứ đều gắn cố định với 30 giây. Clip dài hơn cần chia đoạn; clip ngắn hơn được đệm thêm. Cửa sổ không streaming sẵn — đó là lý do tồn tại của WhisperX, Whisper-Streaming và faster-whisper.

**Chuẩn hoá log-mel.** `(log_mel - mean) / std` với các thống kê lấy từ chính corpus huấn luyện của Whisper. Bạn *bắt buộc* phải dùng tiền xử lý của Whisper (`whisper.audio.log_mel_spectrogram`), không dùng `librosa.feature.melspectrogram`.

### Các biến thể năm 2026

| Biến thể | Tham số | Độ trễ (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Tinh chỉnh

Quy trình chuẩn năm 2026:

1. Thu thập 10–100 giờ audio đúng miền mục tiêu kèm bản chép lời đã căn chỉnh.
2. Chạy `transformers.Seq2SeqTrainer` với callback `generate_with_loss`.
3. Tiết kiệm tham số: low-rank adaptation (LoRA) trên `q_proj`, `k_proj`, `v_proj` của các lớp attention giảm bộ nhớ GPU 4× với mức giá dưới 0,3 WER.
4. Đóng băng encoder nếu bạn có dưới 10 giờ dữ liệu. Chỉ tinh chỉnh decoder.
5. Dùng đúng tokenizer và định dạng prompt của Whisper; đừng bao giờ đổi tokenizer.

Kết quả từ cộng đồng: tinh chỉnh Medium trên 20 giờ đọc chính tả y khoa giảm WER từ 12% xuống 4,5% trên từ vựng y khoa. Tinh chỉnh Turbo trên 4 giờ tiếng Iceland giảm WER từ 18% xuống 6%.

```figure
sp-asr-attention
```

## Tự tay làm

### Bước 1: chạy Whisper ngay mặc định

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Những mặc định quan trọng bạn nên luôn ghi đè: `temperature=0.0` (lấy mẫu mặc định theo chuỗi fallback 0.0 → 0.2 → 0.4 …), `condition_on_previous_text=False` (ngăn vấn đề hallucination (ảo giác) lan truyền), và `no_speech_threshold=0.6` (phát hiện khoảng lặng).

### Bước 2: chia đoạn cho audio dài

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX bổ sung (1) cổng VAD của Silero, (2) căn chỉnh mức từ qua wav2vec 2.0, (3) phân biệt người nói (diarization) qua `pyannote.audio`. Con ngựa thồ của năm 2026 cho chép lời trong môi trường production.

### Bước 3: tinh chỉnh với LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Sau đó là vòng lặp Trainer quen thuộc. Checkpoint mỗi 1000 bước. Đánh giá bằng WER trên tập giữ riêng.

### Bước 4: xem mỗi lớp học được gì

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Trực quan hoá bằng heatmap — bạn sẽ thấy một đường chéo căn chỉnh khi các bước decoder quét qua các frame của encoder. Đường chéo đó chính là cách Whisper biểu diễn mốc thời gian ở mức từ.

## Áp dụng

Bộ công cụ năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Tiếng Anh phổ thông, offline | Large-v3-turbo qua `whisperx` |
| Di động / thiết bị biên | Whisper-Tiny lượng tử hoá (int8) hoặc Moonshine |
| Đa ngôn ngữ, audio dài | Large-v3 qua `whisperx` + diarization |
| Ngôn ngữ ít tài nguyên | Tinh chỉnh Medium hoặc Turbo với LoRA |
| Streaming (độ trễ 2 s) | Whisper-Streaming hoặc Parakeet-TDT |
| Mốc thời gian mức từ | WhisperX (căn chỉnh cưỡng bức qua wav2vec 2.0) |

`faster-whisper` (backend CTranslate2) là runtime suy luận CPU+GPU nhanh nhất năm 2026 — nhanh hơn 4× so với bản gốc mà đầu ra giống hệt.

## Những cạm bẫy vẫn còn trong năm 2026

- **Văn bản ảo giác khi gặp khoảng lặng.** Whisper huấn luyện trên phụ đề nên sinh ra "Thanks for watching!", "Subscribe!", lời bài hát. Luôn chặn bằng VAD trước khi gọi.
- **Hiệu ứng dây chuyền của `condition_on_previous_text`.** Một ảo giác làm bẩn các cửa sổ tiếp theo. Đặt `False` trừ khi bạn cần sự trôi chảy xuyên đoạn.
- **Đệm clip ngắn.** Một clip 2 giây bị đệm thành 30 giây có thể ảo giác trong phần lặng phía sau. Dùng `pad=False` hoặc chặn bằng VAD.
- **Sai thống kê mel.** Dùng mel của librosa thay vì của Whisper cho ra đầu ra gần như ngẫu nhiên. Dùng `whisper.audio.log_mel_spectrogram`.

## Đóng gói

Lưu thành `outputs/skill-whisper-tuner.md`. Thiết kế một pipeline tinh chỉnh hoặc suy luận Whisper cho một miền cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó tokenize một prompt kiểu Whisper, tính ngân sách hình dạng sau giải mã, và in lịch chia đoạn cho một clip 10 phút.
2. **Trung bình.** Cài `faster-whisper`, chép một podcast 10 phút, so sánh WER với bản chép tay của người. Thử `language="auto"` so với ép `language="en"`.
3. **Khó.** Dùng `datasets` của HF, chọn một ngôn ngữ Whisper còn yếu (ví dụ tiếng Urdu), tinh chỉnh Medium bằng LoRA trong 2 epoch trên 2 giờ dữ liệu, và báo cáo mức thay đổi WER.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| 30-sec window | Giới hạn của Whisper | Giới hạn cứng của đầu vào; audio dài hơn phải chia đoạn. |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` mở đầu prompt của decoder. |
| Timestamps token | Căn chỉnh thời gian | Mỗi mốc lệch 0,02 s là một token đặc biệt trong vocab 51k. |
| Turbo | Biến thể nhanh | 4 lớp decoder, nhanh 8×, suy giảm dưới 1% WER. |
| WhisperX | Lớp bọc cho audio dài | VAD + Whisper + căn chỉnh wav2vec + diarization. |
| LoRA fine-tune | Tinh chỉnh hiệu quả | Thêm adapter hạng thấp vào attention; chỉ huấn luyện ~0,3% tham số. |
| Hallucination | Kiểu hỏng âm thầm | Whisper sinh ra tiếng Anh trôi chảy từ nhiễu hoặc khoảng lặng. |

## Đọc thêm

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — kiến trúc và công thức huấn luyện gốc.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — decoder 4 lớp, nhanh 8×.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — audio dài, căn chỉnh mức từ, phân biệt người nói.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — chạy trên CTranslate2, nhanh 4×.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — hướng dẫn chuẩn cho LoRA / full-FT.

---

[← Bài trước: Nhận dạng tiếng nói (ASR)](../../04-speech-recognition-asr/docs/vi.md) · [Bài sau: Nhận dạng & Xác minh người nói →](../../06-speaker-recognition-verification/docs/vi.md)
