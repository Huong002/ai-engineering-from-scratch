# Text-to-Speech (chuyển văn bản thành giọng nói) — Từ Tacotron tới F5 và Kokoro

> ASR đảo giọng nói thành văn bản; TTS đảo văn bản thành giọng nói. Bộ công cụ của 2026 gồm ba phần: văn bản → token, token → mel, mel → waveform. Mỗi phần có một mô hình mặc định vừa với một laptop.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Spectrogram & Mel), Giai đoạn 5, Bài 09 (Mô hình chuỗi sang chuỗi), Giai đoạn 7, Bài 05 (Transformer đầy đủ)
**Time:** ~75 phút

## Vấn đề

Bạn có một chuỗi: "Please remind me to water the plants at 6 pm." Bạn cần một đoạn âm thanh 3 giây nghe tự nhiên, có prosody (ngữ điệu) đúng (chỗ ngắt, trọng âm), phát âm "plants" với nguyên âm chuẩn, và chạy dưới 300 ms trên CPU cho một trợ lý giọng nói trực tiếp. Bạn còn phải đổi được giọng, xử lý đầu vào trộn ngôn ngữ ("remind me at 6 pm, daijoubu?"), và không bị lúng túng trước tên riêng.

Pipeline TTS hiện đại trông như sau:

1. **Text frontend (tiền xử lý văn bản).** Chuẩn hoá văn bản (ngày tháng, số, email), chuyển thành phoneme (âm vị) hoặc token subword, dự đoán các đặc trưng prosody.
2. **Acoustic model (mô hình âm học).** Văn bản → mel spectrogram (phổ mel). Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder (bộ tổng hợp sóng).** Mel → waveform. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), vocoder codec mạng nơ-ron từ 2024 trở đi.

Năm 2026, ranh giới giữa acoustic model và vocoder nhoè đi nhờ các mô hình diffusion và flow-matching đầu-cuối. Nhưng mô hình tinh thần ba phần vẫn hữu ích khi gỡ lỗi.

## Khái niệm

![Tacotron, FastSpeech, VITS, F5/Kokoro đặt cạnh nhau](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: char-embedding → encoder BiLSTM → attention nhạy vị trí → decoder LSTM tự hồi quy phát ra các frame mel. Chậm (AR), rung lắc với văn bản dài. Vẫn được dẫn làm baseline.

**FastSpeech 2 (2020).** Không tự hồi quy. Duration predictor cho biết mỗi phoneme nhận bao nhiêu frame mel. Một lượt, nhanh hơn Tacotron 10×. Mất chút tự nhiên (vì alignment (căn chỉnh) đơn điệu) nhưng có mặt ở khắp nơi.

**VITS (2021).** Huấn luyện chung encoder + duration dựa trên flow + vocoder HiFi-GAN theo kiểu đầu-cuối bằng suy luận biến phân. Chất lượng cao, chỉ một mô hình. Thống trị TTS mã nguồn mở 2022–2024. Biến thể: YourTTS (nhiều giọng, zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Diffusion transformer trên flow matching. Ngữ điệu tự nhiên, nhân bản giọng zero-shot chỉ với 5 giây âm thanh tham chiếu. Đứng đầu các bảng xếp hạng TTS mã nguồn mở năm 2026. 335M tham số.

**Kokoro (2024).** Nhỏ (82M), chạy được trên CPU, TTS tiếng Anh tốt nhất trong phân khúc cho nhu cầu thời gian thực. Từ vựng đóng, chỉ tiếng Anh, apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Đẳng cấp thương mại. Các thẻ cảm xúc ("[whispered]", "[laughing]") và giọng nhân vật của ElevenLabs v2.5 thống trị khâu sản xuất sách nói năm 2026.

### Tiến hoá của vocoder

| Kỷ nguyên | Vocoder | Độ trễ | Chất lượng |
|-----|---------|---------|---------|
| 2016 | WaveNet | chỉ offline | SOTA tại thời điểm phát hành |
| 2018 | WaveRNN | ~thời gian thực | tốt |
| 2020 | HiFi-GAN | 100× thời gian thực | gần mức con người |
| 2022 | BigVGAN | 50× thời gian thực | tổng quát hoá qua nhiều giọng/ngôn ngữ |
| 2024 | SNAC, DAC (codec mạng nơ-ron) | gắn liền với mô hình AR | token rời rạc, tiết kiệm bit |

Tới 2026, phần lớn mô hình "TTS" đã đầu-cuối từ văn bản tới waveform; mel spectrogram chỉ còn là biểu diễn nội bộ.

### Đánh giá

- **MOS (Mean Opinion Score).** Thang 1–5, do đám đông chấm. Vẫn là chuẩn vàng; chậm đến phát mệt.
- **CMOS (Comparative MOS).** So sánh thích A hay B. Khoảng tin cậy hẹp hơn trên mỗi lượt chấm.
- **UTMOS, DNSMOS.** Mô hình nơ-ron dự đoán MOS không cần tham chiếu. Dùng cho bảng xếp hạng.
- **CER (Character Error Rate) qua ASR.** Cho đầu ra TTS chạy qua Whisper, tính CER so với văn bản đầu vào. Đại diện cho độ dễ hiểu.
- **SECS (Speaker Embedding Cosine Similarity).** Chất lượng nhân bản giọng.

Số liệu 2026 trên LibriTTS test-clean:

| Mô hình | UTMOS | CER (qua Whisper) | Kích thước |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## Tự tay làm

### Bước 1: chuyển đầu vào thành phoneme

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phoneme là cây cầu chung. Tránh đưa văn bản thô vào bất cứ thứ gì dưới mức chất lượng của VITS.

### Bước 2: chạy Kokoro (lựa chọn mặc định trên CPU năm 2026)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Chạy offline, một file duy nhất, 82M tham số.

### Bước 3: chạy F5-TTS với nhân bản giọng nói

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Đưa vào một đoạn 5 giây tham chiếu cùng bản ghi lời; F5 sao lại ngữ điệu và âm sắc.

### Bước 4: tự viết vocoder HiFi-GAN

Quá lớn để nhét vào một script hướng dẫn, nhưng hình dạng là:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Huấn luyện: đối kháng (discriminator trên cửa sổ ngắn) + loss tái tạo mel-spectrogram + loss so khớp đặc trưng. Việc này đã phổ thông hoá — hãy dùng checkpoint có sẵn từ repo `hifi-gan` hoặc nvidia-NeMo.

### Bước 5: toàn bộ pipeline (giả mã)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Áp dụng

Bộ công cụ của 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Trợ lý giọng nói tiếng Anh thời gian thực | Kokoro (CPU) hoặc XTTS v2 (GPU) |
| Nhân bản giọng từ 5 giây tham chiếu | F5-TTS |
| Giọng nhân vật thương mại | ElevenLabs v2.5 |
| Đọc sách nói | ElevenLabs v2.5 hoặc XTTS v2 + fine-tune |
| Ngôn ngữ ít tài nguyên | Huấn luyện VITS trên 5–20 h dữ liệu ngôn ngữ đích |
| Giàu biểu cảm / thẻ cảm xúc | ElevenLabs v2.5 hoặc fine-tune StyleTTS 2 |

Dẫn đầu mã nguồn mở tính đến 2026: **F5-TTS về chất lượng, Kokoro về hiệu quả**. Đừng chọn Tacotron trừ khi bạn là người nghiên cứu lịch sử.

## Cạm bẫy

- **Không có bộ chuẩn hoá văn bản.** "Dr. Smith" đọc là "Doctor" hay "Drive"? "2026" là "twenty twenty six" hay "two zero two six"? Chuẩn hoá TRƯỚC phonemizer.
- **Danh từ riêng ngoài từ vựng (OOV).** "Ghumare" → "ghyu-mair"? Hãy có một mô hình grapheme-to-phoneme (từ chữ sang âm vị) dự phòng cho token lạ.
- **Cắt đỉnh (clipping).** Đầu ra vocoder hiếm khi bị cắt, nhưng lệch tỉ lệ mel lúc suy luận có thể vượt ±1.0. Luôn `np.clip(wav, -1, 1)`.
- **Lệch tần số lấy mẫu.** Kokoro xuất 24 kHz; pipeline phía sau của bạn cần 16 kHz → hãy resample, không thì bị aliasing.

## Đóng gói

Lưu thành `outputs/skill-tts-designer.md`. Thiết kế một pipeline TTS cho một mục tiêu cho trước về giọng, độ trễ và ngôn ngữ.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó dựng từ điển phoneme từ một bộ từ vựng đồ chơi, ước lượng thời lượng cho từng phoneme, rồi in ra một lịch "mel" giả.
2. **Trung bình.** Cài Kokoro, tổng hợp cùng một câu với giọng `af_bella` và `am_adam`. So sánh thời lượng âm thanh và cảm nhận chất lượng.
3. **Khó.** Tự ghi một đoạn 5 giây làm tham chiếu. Dùng F5-TTS để nhân bản. Báo cáo SECS giữa tham chiếu và đầu ra được nhân bản.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Phoneme | Đơn vị âm thanh | Lớp âm trừu tượng; tiếng Anh có 39 âm (ARPABet). |
| Duration predictor | Mỗi phoneme kéo dài bao lâu | Đầu ra của mô hình không tự hồi quy; số frame nguyên cho mỗi phoneme. |
| Vocoder | Mel → waveform | Mạng nơ-ron ánh xạ mel-spec sang mẫu thô. |
| HiFi-GAN | Vocoder chuẩn mực | Dựa trên GAN; thống trị 2020–2024. |
| MOS | Chất lượng chủ quan | Điểm ý kiến trung bình thang 1–5 do người chấm. |
| SECS | Chỉ số nhân bản giọng | Cosine similarity giữa embedding người nói đích và đầu ra. |
| F5-TTS | SOTA mã nguồn mở 2024 | Diffusion flow-matching; nhân bản zero-shot. |
| Kokoro | Đầu bảng tiếng Anh trên CPU | Mô hình 82M tham số, Apache 2.0. |

## Đọc thêm

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — baseline seq2seq.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — đầu-cuối dựa trên flow.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — SOTA mã nguồn mở hiện tại.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — vocoder vẫn còn được dùng năm 2026.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — TTS tiếng Anh thân thiện CPU năm 2024.

---

[← Bài trước: Nhận dạng & Xác minh người nói](../../06-speaker-recognition-verification/docs/vi.md) · [Bài sau: Sao chép giọng nói & Chuyển đổi giọng nói →](../../08-voice-cloning-conversion/docs/vi.md)
