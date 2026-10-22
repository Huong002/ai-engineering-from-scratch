# Audio Transformer (transformer cho âm thanh) — Kiến trúc Whisper

> Âm thanh là một tấm ảnh của tần số theo thời gian. Whisper là một ViT ăn spectrogram (phổ đồ) mel rồi đọc lại thành chữ.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 7, Bài 08 (Encoder-decoder), Giai đoạn 7, Bài 09 (ViT)
**Time:** ~45 phút

## Vấn đề

Trước Whisper (OpenAI, Radford et al. 2022), nhận dạng giọng nói tự động (ASR) ở mức tốt nhất là wav2vec 2.0 và HuBERT — bộ trích đặc trưng tự giám sát cộng thêm một head được fine-tune. Chất lượng cao nhưng pipeline dữ liệu đắt đỏ và dễ vỡ khi đổi miền. Nhận dạng giọng nói đa ngôn ngữ thì phải có model riêng cho từng nhóm ngôn ngữ.

Whisper đặt ba cược:

1. **Huấn luyện trên tất cả mọi thứ.** 680.000 giờ âm thanh gán nhãn yếu, thu thập từ internet trên 97 ngôn ngữ. Không có kho dữ liệu học thuật sạch. Không có nhãn âm vị.
2. **Một model cho nhiều tác vụ.** Một decoder được huấn luyện đồng thời trên phiên âm, dịch, phát hiện tiếng nói, nhận dạng ngôn ngữ và gắn timestamp (mốc thời gian) thông qua các task token.
3. **Transformer encoder-decoder (mã hoá-giải mã) tiêu chuẩn.** Encoder tiêu thụ spectrogram log-mel. Decoder sinh token văn bản theo kiểu tự hồi quy. Không vocoder, không CTC, không HMM.

Kết quả: Whisper large-v3 bền bỉ qua các giọng vùng miền, tiếng ồn và những ngôn ngữ không có một giây dữ liệu gán nhãn sạch nào. Nó là front-end giọng nói mặc định cho mọi trợ lý thoại mã nguồn mở, và cả hầu hết trợ lý thương mại, vào 2026.

## Khái niệm

![Pipeline Whisper: âm thanh → mel → encoder → decoder → văn bản](../assets/whisper.svg)

### Bước 1 — resample + cửa sổ

Âm thanh ở 16 kHz. Cắt/đệm về 30 giây. Tính spectrogram log-mel: 80 mel bin, bước 10 ms → ~3.000 frame × 80 đặc trưng. Đây chính là "ảnh đầu vào" mà Whisper nhìn thấy.

### Bước 2 — stem tích chập

Hai lớp Conv1D với kernel 3 và stride 2 rút 3.000 frame xuống 1.500. Giảm một nửa độ dài chuỗi mà không thêm nhiều tham số.

### Bước 3 — encoder

Một transformer encoder 24 lớp (với bản large) trên 1.500 bước thời gian. Mã hoá vị trí dạng sin, self-attention, FFN GELU. Sinh ra 1.500 × 1.280 hidden state.

### Bước 4 — decoder

Một transformer decoder 24 lớp. Nó sinh token tự hồi quy từ bộ từ vựng BPE, vốn là tập cha của GPT-2 cộng thêm vài token đặc biệt dành riêng cho âm thanh.

### Bước 5 — task token

Prompt của decoder bắt đầu bằng các token điều khiển cho model biết phải làm gì:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

hoặc

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Model được huấn luyện trên quy ước này. Bạn điều khiển tác vụ bằng tiền tố. Đây là bản tương đương 2026 của instruction-tuning, nhưng áp lên giọng nói.

### Bước 6 — đầu ra

Beam search (độ rộng 5) với ngưỡng log-prob. Timestamp được dự đoán mỗi 0,02 giây âm thanh khi không có token `<|notimestamps|>`.

### Kích thước các bản Whisper

| Mô hình | Tham số | Số lớp | d_model | Số head | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (decoder 4 lớp) |

Large-v3-turbo (2024) cắt decoder từ 32 lớp xuống 4. Giải mã nhanh gấp 8 lần với mức giảm dưới 1 điểm WER. Chính tốc độ giải mã đó khiến Whisper-turbo thành mặc định cho các voice agent thời gian thực vào 2026.

### Những gì Whisper không làm

- Không tách người nói (diarization — ai đang nói). Ghép với pyannote cho việc đó.
- Không streaming thời gian thực một cách tự nhiên — cửa sổ 30 giây là cố định. Các wrapper hiện đại (`faster-whisper`, `WhisperX`) gắn thêm streaming qua VAD + chồng lấn.
- Không có ngữ cảnh dài hơn 30 giây nếu không cắt khúc từ bên ngoài. Thực tế vẫn chạy tốt vì lời nói của con người hiếm khi cần ngữ cảnh xa để phiên âm.

### Bối cảnh 2026

| Tác vụ | Mô hình | Ghi chú |
|------|-------|-------|
| ASR tiếng Anh | Whisper-turbo, Moonshine | Moonshine nhanh gấp 4 lần trên thiết bị biên |
| ASR đa ngôn ngữ | Whisper-large-v3 | 97 ngôn ngữ |
| ASR streaming | faster-whisper + VAD | Có thể đạt mục tiêu độ trễ 150 ms |
| TTS | Piper, XTTS-v2, Kokoro | Mẫu encoder-decoder, nhưng mang hình dạng Whisper |
| Âm thanh + ngôn ngữ | AudioLM, SeamlessM4T | Token văn bản + token âm thanh trong một transformer |

```figure
n5-mel-decode
```

## Tự tay làm

Xem `code/main.py`. Chúng ta không huấn luyện Whisper — chúng ta dựng pipeline spectrogram log-mel + bộ định dạng prompt bằng task token. Đó là những phần bạn thực sự chạm vào khi làm production.

### Bước 1: tổng hợp âm thanh

Sinh một sóng sin 1 giây ở 440 Hz, lấy mẫu 16 kHz. 16.000 mẫu.

### Bước 2: spectrogram log-mel (rút gọn)

Spectrogram mel đầy đủ cần FFT. Ta làm bản đơn giản hoá gồm chia khung + năng lượng từng khung, đủ để thấy pipeline mà không cần `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms, hop = 10 ms. Khớp với cách Whisper chia cửa sổ. Năng lượng từng khung thay cho mel bin để dễ học.

### Bước 3: đệm lên 30 giây

Whisper luôn xử lý từng khúc 30 giây. Đệm (hoặc cắt) spectrogram về 3.000 frame.

### Bước 4: dựng các token prompt

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Đó là toàn bộ bề mặt điều khiển tác vụ. Một tiền tố 4 token.

## Áp dụng

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Nhanh hơn, tương thích OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Khi nào nên chọn Whisper vào 2026:**

- ASR đa ngôn ngữ với một model duy nhất.
- Phiên âm bền bỉ với âm thanh nhiễu và đa dạng.
- Nghiên cứu / làm thử ASR — điểm khởi đầu nhanh nhất.

**Khi nào nên chọn thứ khác:**

- Streaming độ trễ cực thấp trên thiết bị biên — Moonshine thắng Whisper ở cùng chất lượng.
- AI hội thoại thời gian thực cần dưới 200 ms — dùng ASR streaming chuyên dụng.
- Tách người nói — Whisper không làm việc này; hãy gắn thêm pyannote.

## Đóng gói

Xem `outputs/skill-asr-configurator.md`. Skill này chọn model ASR, tham số giải mã và pipeline tiền xử lý cho một ứng dụng giọng nói mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác nhận số frame của tín hiệu 1 giây ở 16 kHz với hop 10 ms là ~100 frame. Với 30 giây: ~3.000 frame.
2. **Trung bình.** Dựng spectrogram log-mel đầy đủ bằng `numpy.fft`. Kiểm tra 80 mel bin khớp với `librosa.feature.melspectrogram(n_mels=80)` trong sai số số học.
3. **Khó.** Cài đặt suy luận streaming: cắt âm thanh thành các cửa sổ 10 giây chồng lấn 2 giây, chạy Whisper trên từng khúc, gộp các bản phiên âm. Đo word-error rate so với chạy một lượt trên một mẫu podcast 5 phút.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Mel spectrogram | "Ảnh âm thanh" | Biểu diễn 2D: trục này là các bin tần số, trục kia là các frame thời gian; năng lượng mỗi ô theo thang log. |
| Log-mel | "Thứ Whisper nhìn thấy" | Spectrogram mel đưa qua hàm log; xấp xỉ cách con người cảm nhận độ to. |
| Frame | "Một lát thời gian" | Một cửa sổ 25 ms mẫu; chồng lấn với bước 10 ms. |
| Task token | "Tiền tố prompt cho giọng nói" | Token đặc biệt như `<\|transcribe\|>` / `<\|translate\|>` trong prompt của decoder. |
| Voice activity detection (VAD) | "Tìm chỗ có tiếng nói" | Cổng loại bỏ khoảng lặng trước khi đưa vào ASR; cắt chi phí rất mạnh. |
| CTC | "Connectionist Temporal Classification" | Loss kinh điển của ASR để huấn luyện không cần căn chỉnh; Whisper KHÔNG dùng nó. |
| Whisper-turbo | "Decoder nhỏ, encoder đầy đủ" | Encoder large-v3 + decoder 4 lớp; giải mã nhanh gấp 8 lần. |
| Faster-whisper | "Wrapper production" | Bản cài lại trên CTranslate2; lượng tử int8; nhanh gấp 4 lần bản tham chiếu của OpenAI. |

## Đọc thêm

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — bài báo Whisper.
- [Repo OpenAI Whisper](https://github.com/openai/whisper) — code tham chiếu + trọng số model. Đọc `whisper/model.py` để thấy stem Conv1D + encoder + decoder từ trên xuống dưới trong ~400 dòng.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — logic beam-search + task token mô tả ở Bước 5–6 nằm ở đây; 500 dòng, đọc trọn được.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — tiền thân; vẫn cho đặc trưng SOTA trong một số bối cảnh.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — wrapper production, nhanh gấp 4 lần bản tham chiếu.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — ASR thân thiện thiết bị biên 2024, cùng hình dạng Whisper nhưng nhỏ hơn.
- [Blog HuggingFace — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — công thức fine-tuning chuẩn, gồm bộ tiền xử lý spectrogram mel và cách xử lý token-timestamp.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — bản cài đặt đầy đủ (encoder, decoder, cross-attention, generation) khớp với sơ đồ kiến trúc của bài học.

---

[← Bài trước: Vision Transformers (ViT)](../../09-vision-transformers/docs/vi.md) · [Bài sau: Mixture of Experts (MoE) →](../../11-mixture-of-experts/docs/vi.md)
