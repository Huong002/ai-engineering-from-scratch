# Kiến thức nền về âm thanh — Dạng sóng, lấy mẫu, biến đổi Fourier

> Dạng sóng (waveform) là tín hiệu thô. Spectrogram là biểu diễn. Đặc trưng Mel là dạng thân thiện với ML. Mọi pipeline ASR và TTS hiện đại đều leo thang này, và nấc đầu tiên là hiểu lấy mẫu và Fourier.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 1, Bài 06 (Vector & Ma trận), Giai đoạn 1, Bài 14 (Phân phối xác suất)
**Time:** ~45 phút

## Vấn đề

Microphone tạo ra tín hiệu áp suất theo thời gian. Mạng nơ-ron của bạn tiêu thụ tensor. Giữa hai bên là một chồng quy ước mà khi bị vi phạm sẽ sinh ra những lỗi âm thầm: model train tốt nhưng WER tăng gấp đôi, hoặc TTS xuất xưởng kèm tiếng rít, hoặc hệ thống nhân bản giọng nói học thuộc cái microphone thay vì người nói.

Mọi lỗi trong hệ thống giọng nói đều quy về một trong ba câu hỏi:

1. Dữ liệu được ghi ở tần số lấy mẫu nào, và model mong đợi cái gì?
2. Tín hiệu có bị chồng phổ không?
3. Bạn đang thao tác trên mẫu thô hay trên một biểu diễn tần số?

Làm đúng ba điều này thì phần còn lại của Giai đoạn 6 là chuyện giải được. Làm sai thì đến cả Whisper-Large-v4 cũng cho ra rác.

## Khái niệm

![Dạng sóng, lấy mẫu, DFT và các bin tần số được trực quan hoá](../assets/audio-fundamentals.svg)

**Waveform.** Một mảng một chiều các số thực trong `[-1.0, 1.0]`. Đánh chỉ số theo số mẫu. Để đổi sang giây, chia cho tần số lấy mẫu (sample rate): `t = n / sr`. Một clip 10 giây ở 16 kHz là một mảng 160,000 số thực.

**Tần số lấy mẫu (sr).** Số mẫu trong một giây. Những mức phổ biến năm 2026:

| Tốc độ | Công dụng |
|------|-----|
| 8 kHz | Điện thoại, VOIP cũ. Nyquist ở 4 kHz giết phụ âm. Tránh dùng cho ASR. |
| 16 kHz | Chuẩn ASR. Whisper, Parakeet, SeamlessM4T v2 đều dùng 16 kHz. |
| 22.05 kHz | Train vocoder TTS cho các model đời cũ. |
| 24 kHz | TTS hiện đại (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | Audio CD, nhạc. |
| 48 kHz | Phim, pro audio, TTS độ trung thực cao (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** Một tần số lấy mẫu `sr` có thể biểu diễn không nhập nhằng các tần số tới `sr/2`. Ranh giới `sr/2` là *tần số Nyquist*. Năng lượng phía trên Nyquist bị *aliased* (hiện tượng chồng phổ) — gấp xuống các tần số thấp hơn — và làm hỏng tín hiệu. Luôn lọc low-pass trước khi hạ tần số lấy mẫu.

**Bit depth.** 16-bit PCM (int16 có dấu, khoảng ±32,767) là định dạng trao đổi phổ quát. 24-bit cho nhạc, 32-bit float cho DSP nội bộ. Các thư viện như `soundfile` đọc int16 nhưng đưa ra mảng float32 trong `[-1, 1]`.

**Fourier Transform.** Mọi tín hiệu hữu hạn là tổng của các hàm sin ở những tần số (frequency) khác nhau. Biến đổi Fourier rời rạc (DFT) tính, với `N` mẫu, `N` hệ số phức — mỗi hệ số ứng với một bin tần số. `bin k` ánh xạ tới tần số `k · sr / N` Hz. Độ lớn là biên độ (amplitude) tại tần số đó, góc là pha.

**FFT.** Fast Fourier Transform: thuật toán `O(N log N)` cho DFT khi `N` là lũy thừa của 2. Mọi thư viện âm thanh đều dùng FFT bên dưới. Một FFT 1024 mẫu ở 16 kHz cho 512 bin tần số dùng được, trải từ 0–8 kHz với độ phân giải 15.6 Hz.

**Framing + window.** Ta không FFT cả một clip. Ta chặt nó thành các *frame* chồng lấn (thường 25 ms với hop 10 ms), nhân mỗi frame với một hàm cửa sổ (Hann, Hamming) để diệt các gián đoạn ở biên, rồi FFT từng frame. Đó là Short-Time Fourier Transform (STFT). Bài 02 tiếp tục từ đây.

```figure
mel-scale
```

## Tự tay làm

### Bước 1: đọc một clip và vẽ dạng sóng

`code/main.py` chỉ dùng module `wave` của stdlib để bản demo không phụ thuộc gì. Khi làm thật bạn sẽ dùng `soundfile` hoặc `torchaudio.load` (cả hai đều trả về tuple `(waveform, sr)`):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Bước 2: tổng hợp sóng sin từ nguyên lý gốc

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Một sóng sin 440 Hz (nốt La chuẩn) ở 16 kHz trong 1 giây là 16,000 số thực. Ghi bằng `wave.open(..., "wb")` với mã hoá 16-bit PCM.

### Bước 3: tự tay tính DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — ổn với `N=256` để xác nhận tính đúng, vô dụng với audio thật. Code thật gọi `numpy.fft.rfft` hoặc `torch.fft.rfft`.

### Bước 4: tìm tần số trội

Chỉ số đỉnh độ lớn `k_star` ánh xạ tới tần số `k_star * sr / N`. Chạy trên sóng sin 440 Hz sẽ cho đỉnh tại bin `440 * N / sr`.

### Bước 5: minh hoạ hiện tượng chồng phổ

Lấy mẫu một sóng sin 7 kHz ở 10 kHz (Nyquist = 5 kHz). Âm 7 kHz nằm trên Nyquist nên gấp về `10 − 7 = 3 kHz`. Đỉnh FFT xuất hiện ở 3 kHz. Đây là màn demo chồng phổ kinh điển và là lý do mọi DAC/ADC đều có sẵn bộ lọc low-pass dựng tường.

## Áp dụng

Bộ công cụ bạn thực sự dùng để xuất xưởng năm 2026:

| Việc | Thư viện | Lý do |
|------|---------|-----|
| Đọc/ghi WAV/FLAC/OGG | `soundfile` (wrapper libsndfile) | Nhanh nhất, ổn định, trả về float32. |
| Đổi tần số lấy mẫu | `torchaudio.transforms.Resample` hoặc `librosa.resample` | Chống chồng phổ đúng cách, có sẵn. |
| STFT / Mel | `torchaudio` hoặc `librosa` | Thân thiện GPU; hệ sinh thái PyTorch. |
| Truyền phát thời gian thực | `sounddevice` hoặc `pyaudio` | Binding PortAudio đa nền tảng. |
| Xem thông tin file | `ffprobe` hoặc `soxi` | CLI, nhanh, báo cáo sr/channels/codec. |

Nguyên tắc quyết định: **khớp tần số lấy mẫu trước khi khớp bất cứ thứ gì khác**. Whisper mong đợi 16 kHz mono float32. Đưa cho nó 44.1 kHz stereo thì bạn sẽ nhận về rác trông như lỗi model.

## Đóng gói

Lưu thành `outputs/skill-audio-loader.md`. Skill này giúp bạn kiểm tra rằng đầu vào âm thanh khớp với kỳ vọng của model phía sau và hạ tần số lấy mẫu đúng cách khi không khớp.

## Bài tập

1. **Dễ.** Tổng hợp một bản mix 1 giây gồm 220 Hz + 440 Hz + 880 Hz ở 16 kHz. Chạy DFT. Xác nhận ba đỉnh ở các bin mong đợi.
2. **Trung bình.** Ghi một file WAV 3 giây giọng bạn ở 48 kHz. Hạ xuống 16 kHz bằng `torchaudio.transforms.Resample` (có chống chồng phổ), rồi hạ xuống 16 kHz bằng cách bỏ mẫu thô (lấy mỗi mẫu thứ ba). FFT cả hai. Hiện tượng chồng phổ xuất hiện ở đâu?
3. **Khó.** Tự xây STFT chỉ bằng `math` và hàm DFT ở Bước 3. Frame size 400, hop 160, cửa sổ Hann. Vẽ độ lớn bằng `matplotlib.pyplot.imshow`. Đây chính là spectrogram của Bài 02.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Sample rate | Số mẫu mỗi giây | Tần số tính bằng Hz mà tại đó ADC đo tín hiệu. |
| Nyquist | Tần số lớn nhất có thể biểu diễn | `sr/2`; năng lượng phía trên bị chồng phổ ngược xuống. |
| Bit depth | Độ phân giải của mỗi mẫu | `int16` = 65,536 mức; `float32` = độ chính xác 24-bit trong `[-1, 1]`. |
| DFT | Biến đổi Fourier cho dãy | `N` mẫu → `N` hệ số tần số phức. |
| FFT | Bản DFT nhanh | Thuật toán `O(N log N)` đòi hỏi `N` = lũy thừa của 2. |
| Bin | Cột tần số | `k · sr / N` Hz; độ phân giải = `sr / N`. |
| STFT | Spectrogram ở bên dưới | FFT có chia frame + cửa sổ, chạy theo thời gian. |
| Aliasing | Bóng ma tần số kỳ lạ | Năng lượng trên Nyquist soi gương xuống các bin thấp hơn. |

## Đọc thêm

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — bài báo đằng sau định lý lấy mẫu.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — sách giáo khoa DSP miễn phí, kinh điển.
- [librosa docs — audio primer](https://librosa.org/doc/latest/auto_tutorials/index.html) — hướng dẫn thực hành kèm code.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.taylorfrancis.com/books/mono/10.1201/9781315372150/room-acoustics-heinrich-kuttruff) — tài liệu tham khảo về lý do audio thực tế không phải sóng sin sạch.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — trực giác về bin tần số được làm rõ trong 10 phút.

---

[Bài sau: Phổ đồ, thang mel & đặc trưng âm thanh →](../../02-spectrograms-mel-features/docs/vi.md)
