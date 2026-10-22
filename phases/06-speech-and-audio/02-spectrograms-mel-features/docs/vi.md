# Phổ đồ, thang mel & đặc trưng âm thanh

> Mạng nơ-ron không tiếp nhận dạng sóng thô tốt. Chúng tiếp nhận spectrogram (phổ đồ). Chúng còn tiếp nhận mel spectrogram tốt hơn nữa. Mọi hệ ASR, TTS và bộ phân loại âm thanh năm 2026 đều sống còn nhờ đúng một lựa chọn tiền xử lý này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 01 (Nền tảng âm thanh)
**Time:** ~45 phút

## Vấn đề

Lấy một clip 10 giây, 16 kHz. Đó là 160,000 số thực, đều nằm trong `[-1, 1]`, gần như hoàn toàn không tương quan với nhãn "chó sủa" hay "từ mèo". Dạng sóng thô có thông tin, nhưng ở dạng mà mô hình không dễ trích ra. Hai âm vị giống hệt nhau nói cách nhau 100 ms có mẫu thô hoàn toàn khác nhau.

Spectrogram giải quyết chuyện này. Nó nén bớt chi tiết thời gian ở chỗ tri giác con người bỏ qua (rung vi giây) và giữ lại cấu trúc ở chỗ tri giác chú ý (tần số nào đang có năng lượng, trên những cửa sổ thời gian ~10–25 ms).

Mel spectrogram còn đi xa hơn. Con người cảm nhận cao độ theo thang logarit: 100 Hz so với 200 Hz nghe "cách nhau cùng khoảng" như 1000 Hz so với 2000 Hz. Thang mel bẻ cong trục tần số cho khớp với điều đó. Mel spectrogram là đặc trưng quan trọng nhất trong học máy về tiếng nói từ 2010 đến 2026.

## Khái niệm

![Thang bậc từ dạng sóng đến STFT đến mel spectrogram đến MFCC](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform, tức biến đổi Fourier thời gian ngắn).** Cắt dạng sóng thành các frame (khung) chồng lấp (thường gặp: window (cửa sổ) 25 ms, hop length (bước nhảy) 10 ms = 400 mẫu / 160 mẫu ở 16 kHz). Nhân từng khung với một hàm cửa sổ (Hann là mặc định; Hamming đánh đổi hơi khác). FFT từng khung. Xếp các phổ biên độ thành một ma trận có shape `(n_frames, n_freq_bins)`. Đó chính là spectrogram của bạn.

**Biên độ log.** Biên độ thô trải trên 5-6 bậc độ lớn. Lấy `log(|X| + 1e-6)` hoặc `20 * log10(|X|)` để nén dải động. Mọi pipeline thực tế đều dùng biên độ log, không dùng biên độ thô.

**Mel scale (thang mel).** Tần số `f` tính bằng Hz ánh xạ sang mel `m` theo `m = 2595 * log10(1 + f / 700)`. Ánh xạ này gần như tuyến tính dưới 1 kHz và gần như logarit phía trên. 80 bin mel phủ 0–8 kHz là đầu vào ASR tiêu chuẩn.

**Mel filterbank (băng lọc mel).** Một tập bộ lọc tam giác đặt cách đều nhau trên thang mel. Mỗi bộ lọc là tổng có trọng số của các bin FFT kề nhau. Nhân biên độ STFT với ma trận băng lọc cho ra mel spectrogram chỉ trong một phép matmul.

**Spectrogram log-mel.** `log(mel_spec + 1e-10)`. Đầu vào của Whisper. Đầu vào của Parakeet. Đầu vào của SeamlessM4T. Frontend âm thanh phổ quát của năm 2026.

**MFCC.** Lấy spectrogram log-mel, áp một DCT (type II), giữ 13 hệ số đầu. Cách này khử tương quan giữa các đặc trưng và nén thêm. Đây là đặc trưng thống trị cho tới khoảng 2015, khi CNN/Transformer trên log-mel thô bắt kịp. Vẫn còn dùng trong nhận dạng người nói (x-vectors, ECAPA).

**Đánh đổi độ phân giải.** FFT lớn hơn = phân giải tần số tốt hơn nhưng phân giải thời gian kém hơn. 25 ms / 10 ms là mặc định của học máy âm thanh; 50 ms / 12.5 ms cho âm nhạc; 5 ms / 2 ms để phát hiện âm chớp (tiếng gõ trống, phụ âm bật).

```figure
spectrogram-window
```

## Tự tay làm

### Bước 1: chia khung dạng sóng

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

Một clip 10 giây 16 kHz với `frame_len=400, hop=160` cho ra 998 khung.

### Bước 2: cửa sổ Hann

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Nhân từng phần tử trước khi FFT. Bước này loại bỏ rò rỉ phổ do việc cắt cụt tại các điểm cuối khác 0.

### Bước 3: biên độ STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Môi trường thực tế dùng `torch.stft` hoặc `librosa.stft` (dựa trên FFT, đã vector hoá). Vòng lặp ở đây mang tính minh hoạ; nó chạy trên các clip ngắn trong `code/main.py`.

### Bước 4: băng lọc mel

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mel phủ 0–8 kHz với `n_fft=400` cho một ma trận `(80, 201)`. Nhân biên độ STFT `(n_frames, 201)` với chuyển vị của nó để được mel spectrogram `(n_frames, 80)`.

### Bước 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Các lựa chọn thay thế phổ biến: `librosa.power_to_db` (dB chuẩn hoá theo tham chiếu), `10 * log10(power + eps)`. Whisper dùng một quy trình cắt + chuẩn hoá phức tạp hơn (xem `log_mel_spectrogram` của Whisper).

### Bước 6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Áp DCT cho từng khung log-mel, giữ 13 hệ số đầu. Đó là ma trận MFCC của bạn. Hệ số đầu thường bị bỏ (nó mã hoá năng lượng tổng thể).

## Áp dụng

Bộ công cụ của năm 2026:

| Tác vụ | Đặc trưng |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mel, bước nhảy 10 ms, cửa sổ 25 ms |
| Mô hình acoustic cho TTS (VITS, F5-TTS, Kokoro) | 80 mel, bước nhảy 5–12 ms để kiểm soát thời gian mịn |
| Phân loại âm thanh (AST, PANNs, BEATs) | 128 log-mel, bước nhảy 10 ms |
| Embedding người nói (ECAPA-TDNN, WavLM) | 80 log-mel hoặc SSL trên dạng sóng thô |
| Âm nhạc (MusicGen, Stable Audio 2) | token rời rạc EnCodec (không phải mel) |
| Phát hiện từ khoá | 40 MFCC cho thiết bị nhỏ |

Quy tắc ngón tay cái: **nếu bạn không làm về âm nhạc, hãy bắt đầu với 80 log-mel.** Gánh nặng chứng minh thuộc về mọi phương án lệch khỏi mặc định đó.

## Những cạm bẫy vẫn còn được tung ra năm 2026

- **Lệch số lượng mel.** Huấn luyện với 80 mel, suy luận với 128 mel. Hỏng âm thầm. Hãy log shape của đặc trưng ở cả hai đầu.
- **Lệch tần số lấy mẫu ở phía trước.** Mel tính ở 22.05 kHz trông khác với ở 16 kHz. Sửa SR *trước* khi trích đặc trưng.
- **dB so với log.** Whisper chờ log-mel, không phải dB-mel. Một số pipeline HF tự dò; code tuỳ biến của bạn thì không.
- **Trôi chuẩn hoá.** Chuẩn hoá theo từng phát ngôn khi huấn luyện, chuẩn hoá toàn cục khi suy luận. Lỗi thực tế làm WER tăng gấp đôi.
- **Rò rỉ do padding.** Đệm 0 ở cuối clip tạo ra phổ phẳng ở các khung cuối. Hãy đệm đối xứng hoặc lặp lại.

## Đóng gói

Lưu thành `outputs/skill-feature-extractor.md`. Skill này chọn loại đặc trưng, số mel, frame/hop và cách chuẩn hoá cho một mô hình đích cho trước.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Chương trình tổng hợp một chirp (tần số quét 200 → 4000 Hz) và in ra bin mel có argmax theo từng khung. Vẽ đồ thị (tuỳ chọn) và xác nhận nó khớp với đường quét.
2. **Trung bình.** Chạy lại với `n_mels` trong `{40, 80, 128}` và `frame_len` trong `{200, 400, 800}`. Đo băng thông của đỉnh nhọn dọc trục thời gian. Tổ hợp nào phân giải chirp tốt nhất?
3. **Khó.** Cài `power_to_db` và so sánh độ chính xác ASR của một bộ phân loại CNN nhỏ trên AudioMNIST dùng (a) log-mel thô, (b) dB-mel với `ref=max`, (c) MFCC-13 + delta + delta-delta. Báo cáo độ chính xác top-1.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Frame | Một lát cắt | Khúc dạng sóng 25 ms được đưa vào một FFT. |
| Hop | Sải bước | Số mẫu giữa hai khung liên tiếp; 10 ms là mặc định của ASR. |
| Window | Thứ Hann/Hamming | Hệ số nhân từng điểm, làm hai mép khung thu về 0. |
| STFT | Máy tạo spectrogram | FFT trên khung đã chia và đã nhân cửa sổ; cho ra ma trận thời gian × tần số. |
| Mel | Tần số bị bẻ cong | Thang tri giác logarit; `m = 2595·log10(1 + f/700)`. |
| Filterbank | Cái ma trận | Các bộ lọc tam giác chiếu STFT lên các bin mel. |
| Log-mel | Đầu vào của Whisper | `log(mel_spec + eps)`; đã thành chuẩn từ 2026. |
| MFCC | Đặc trưng kiểu cũ | DCT của log-mel; 13 hệ số, đã khử tương quan. |

## Đọc thêm

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — bài báo MFCC.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — thang mel nguyên bản.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — đọc bản cài đặt tham chiếu.
- [librosa feature extraction docs](https://librosa.org/doc/latest/api/feature.html) — tài liệu tham chiếu cho `mfcc`, `melspectrogram`, và hop/window.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — pipeline quy mô thực tế cho các mô hình Parakeet + Canary.

---

[← Bài trước: Kiến thức nền về âm thanh](../../01-audio-fundamentals/docs/vi.md) · [Bài sau: Phân loại âm thanh →](../../03-audio-classification/docs/vi.md)
