# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC và sự tách semantic-acoustic

> Sinh âm thanh năm 2026 gần như toàn là token. EnCodec, SNAC, Mimi và DAC biến dạng sóng liên tục thành chuỗi rời rạc để transformer có thể dự đoán. Cách tách token semantic và token acoustic — sổ mã đầu là semantic, phần còn lại là acoustic — là bước chuyển kiến trúc quan trọng nhất trong âm thanh kể từ Transformer.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Spectrogram), Giai đoạn 10, Bài 11 (Lượng tử hoá), Giai đoạn 5, Bài 19 (Token hoá subword)
**Time:** ~60 phút

## Vấn đề

Mô hình ngôn ngữ làm việc trên token rời rạc. Âm thanh thì liên tục. Nếu bạn muốn một mô hình kiểu LLM cho lời nói / âm nhạc — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — trước tiên bạn cần một **neural audio codec** (codec âm thanh nơ-ron): một encoder học được, rời rạc hoá âm thanh thành bộ từ vựng token nhỏ, cùng một decoder tương ứng tái tạo lại dạng sóng.

Hai dòng codec đã xuất hiện:

1. **Reconstruction-first codecs** (codec ưu tiên tái tạo) — EnCodec, DAC. Tối ưu chất lượng âm thanh theo cảm nhận. Token là "acoustic" — chúng nắm mọi thứ, gồm danh tính người nói, âm sắc, tiếng ồn nền.
2. **Semantic-first codecs** (codec ưu tiên semantic) — Mimi (Kyutai), SpeechTokenizer. Buộc sổ mã (codebook) đầu tiên mã hoá nội dung ngôn ngữ / ngữ âm (thường bằng cách chưng cất từ WavLM). Các sổ mã sau là chi tiết acoustic.

Bài học 2024-2026: **một codec thuần tái tạo sẽ cho bạn lời nói mờ nhoè khi bạn cố sinh ra từ văn bản.** LLM trên token của codec phải học cả cấu trúc ngôn ngữ VÀ cấu trúc acoustic trong cùng một sổ mã, và điều đó không mở rộng được. Tách chúng ra — sổ mã semantic 0, các sổ mã acoustic 1-N — là thứ làm Moshi và Sesame CSM hoạt động.

## Khái niệm

![Bốn codec trong bức tranh tổng thể: EnCodec, DAC, SNAC (đa tỉ lệ), Mimi (semantic + acoustic)](../assets/codec-comparison.svg)

### Mánh khoá cốt lõi: Residual Vector Quantization (RVQ)

Thay vì một sổ mã lớn (muốn chất lượng tốt thì sẽ cần hàng triệu mã), mọi codec âm thanh hiện đại đều dùng **RVQ**: một tầng thác các sổ mã nhỏ. Sổ mã đầu tiên lượng tử hoá đầu ra của encoder; sổ mã thứ hai lượng tử hoá phần dư (residual); v.v. Mỗi sổ mã có 1024 mã. 8 sổ mã = từ vựng hiệu dụng 1024^8 = 10^24.

Khi suy luận, decoder cộng tất cả mã được chọn trong mỗi khung để tái tạo (reconstruction) tín hiệu.

### Bốn codec quan trọng trong năm 2026

**EnCodec (Meta, 2022).** Đường cơ sở. Encoder-decoder trên dạng sóng, nút thắt RVQ. 24 kHz, có thể tới 32 sổ mã, mặc định 4 sổ mã @ 1.5 kbps. Dùng kiến trúc `1D conv + transformer + 1D conv`. Được MusicGen dùng.

**DAC (Descript, 2023).** RVQ với sổ mã chuẩn hoá L2, hàm kích hoạt tuần hoàn, các loss cải tiến. Độ trung thực tái tạo cao nhất trong mọi codec mở — có lúc không phân biệt được với lời nói gốc khi dùng 12 sổ mã. 44.1 kHz toàn dải.

**SNAC (Hubert Siuzdak, 2024).** RVQ đa tỉ lệ — các sổ mã thô chạy ở tốc độ khung thấp hơn các sổ mã mịn. Mô hình hoá âm thanh theo thứ bậc: một "bản phác" thô ở ~12 Hz cộng chi tiết ở 50 Hz. Được Orpheus-3B dùng vì cấu trúc thứ bậc khớp tốt với cách sinh bằng LM.

**Mimi (Kyutai, 2024).** Kẻ thay đổi cuộc chơi năm 2026. Tốc độ khung 12.5 Hz (cực thấp), 8 sổ mã @ 4.4 kbps. Sổ mã 0 được **chưng cất từ WavLM** — huấn luyện để dự đoán đặc trưng nội dung lời nói của WavLM. Sổ mã 1-7 là phần dư acoustic. Cách tách này là nền cho Moshi (Bài 15) và Sesame CSM.

### Tốc độ khung quan trọng với mô hình ngôn ngữ

Tốc độ khung thấp hơn = chuỗi ngắn hơn = LM nhanh hơn.

| Codec | Tốc độ khung | 1 s = N khung | Dùng tốt cho |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | âm nhạc, âm thanh tổng quát |
| DAC-44.1k | 86 Hz | 86 | âm nhạc độ trung thực cao |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM hiệu quả |
| Mimi | 12.5 Hz | 12.5 | lời nói streaming |

Ở 12.5 Hz, một câu nói 10 giây chỉ là 125 khung codec — transformer dễ dàng dự đoán chúng.

### Token semantic và token acoustic

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Token semantic (sổ mã 0 trong Mimi).** Mã hoá điều được nói — âm vị, từ, nội dung. Chưng cất từ WavLM qua một loss dự đoán phụ.
- **Token acoustic (sổ mã 1-7).** Mã hoá âm sắc, danh tính người nói, ngữ điệu, tiếng ồn nền, chi tiết mịn.

Một AR LM dự đoán token semantic trước (điều kiện hoá trên văn bản), rồi dự đoán các token acoustic (điều kiện hoá trên semantic + mẫu giọng tham chiếu). Cách phân rã này là lý do TTS hiện đại nhân bản được giọng zero-shot: mô hình semantic lo nội dung; mô hình acoustic lo âm sắc.

### Chất lượng tái tạo năm 2026 (bit trên giây, tốc độ bit (bitrate) càng thấp càng tốt)

| Codec | Tốc độ bit | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Codec truyền thống như Opus vẫn thắng trên mỗi bit về chất lượng cảm nhận. Codec nơ-ron thắng ở **token rời rạc** (thứ Opus không tạo ra) và **chất lượng cho mô hình sinh** (điều LM có thể làm với các token đó).

```figure
rvq-codec-cascade
```

## Tự tay làm

### Bước 1: mã hoá với EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` ở 6 kbps — tức băng thông (bandwidth) mục tiêu. Mỗi mã là 0-1023 (10-bit).

### Bước 2: giải mã và đo chất lượng tái tạo

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Bước 3: tách semantic-acoustic (kiểu Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Sổ mã semantic 0 căn chỉnh với WavLM. Bạn có thể huấn luyện một transformer text-to-semantic — từ vựng nhỏ hơn nhiều so với đi thẳng sang audio. Sau đó một decoder acoustic-to-waveform riêng điều kiện hoá trên mẫu giọng tham chiếu.

### Bước 4: vì sao AR LM trên token codec lại hiệu quả

Với một đoạn lời nói 10 s ở Mimi 12.5 Hz × 8 sổ mã:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 token là ngữ cảnh tầm thường với transformer. Một transformer 256M tham số có thể sinh 10 giây lời nói trong vài mili giây trên GPU hiện đại.

## Áp dụng

Ánh xạ bài toán → codec:

| Việc | Codec |
|------|-------|
| Sinh nhạc tổng quát | EnCodec-24k |
| Tái tạo trung thực nhất | DAC-44.1k |
| AR LM trên lời nói (TTS) | SNAC hoặc Mimi |
| Lời nói song công streaming | Mimi (12.5 Hz) |
| Thư viện hiệu ứng âm thanh kèm văn bản | EnCodec + điều kiện T5 |
| Biên tập âm thanh chi tiết | DAC + inpainting |

Quy tắc ngón tay cái: **nếu bạn xây mô hình sinh, hãy bắt đầu với Mimi hoặc SNAC. Nếu bạn xây pipeline nén, hãy dùng Opus.**

## Cạm bẫy

- **Quá nhiều sổ mã.** Thêm sổ mã tăng độ trung thực tuyến tính nhưng cũng tăng độ dài chuỗi của LM tuyến tính. Dừng ở 8-12.
- **Lệch tốc độ khung.** Huấn luyện LM trên Mimi 12.5 Hz rồi fine-tune trên EnCodec 50 Hz sẽ hỏng một cách âm thầm.
- **Cho rằng mọi sổ mã như nhau.** Trong Mimi, sổ mã 0 mang nội dung; mất nó là mất khả năng nghe hiểu. Mất sổ mã 7 gần như không nhận ra.
- **Chỉ dùng chất lượng tái tạo làm thước đo.** Một codec có thể tái tạo rất tốt nhưng vô dụng cho sinh bằng LM nếu cấu trúc semantic tệ.

## Đóng gói

Lưu thành `outputs/skill-codec-picker.md`. Chọn một codec cho một bài toán sinh hoặc nén cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó cài đặt một bộ lượng tử hoá vô hướng + phần dư đồ chơi và đo lỗi tái tạo khi bạn thêm sổ mã.
2. **Trung bình.** Cài `encodec` và so sánh 1, 4, 8, 32 sổ mã trên một đoạn lời nói giữ lại. Vẽ PESQ hoặc MSE theo tốc độ bit.
3. **Khó.** Nạp Mimi. Mã hoá một đoạn. Thay sổ mã 0 bằng các số nguyên ngẫu nhiên; giải mã. Rồi thay sổ mã 7 tương tự. So sánh hai kiểu làm hỏng — làm hỏng sổ mã 0 phải phá khả năng nghe hiểu; làm hỏng sổ mã 7 gần như chẳng đổi gì.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| RVQ | Lượng tử hoá phần dư | Thác các sổ mã nhỏ; mỗi sổ mã lượng tử hoá phần dư trước đó. |
| Frame rate | Tốc độ codec | Bao nhiêu khung token mỗi giây. Thấp hơn = LM nhanh hơn. |
| Semantic codebook | Sổ mã 0 (Mimi) | Sổ mã chưng cất từ đặc trưng SSL; mã hoá nội dung. |
| Acoustic codebooks | Mọi thứ còn lại | Âm sắc, ngữ điệu, tiếng ồn, chi tiết mịn. |
| PESQ / ViSQOL | Chất lượng cảm nhận | Các thước đo khách quan tương quan với MOS. |
| EnCodec | Codec của Meta | Đường cơ sở RVQ; được MusicGen dùng. |
| Mimi | Codec của Kyutai | Tốc độ khung 12.5 Hz; tách semantic-acoustic; là nền cho Moshi. |

## Đọc thêm

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — đường cơ sở RVQ.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — codec mở có độ trung thực cao nhất.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — RVQ đa tỉ lệ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — tách semantic-acoustic, chưng cất WavLM.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — mô hình hai giai đoạn semantic/acoustic.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec RVQ truyền phát đầu tiên.

---

[← Bài trước: Xây dựng quy trình xử lý trợ lý giọng nói](../../12-voice-assistant-pipeline/docs/vi.md) · [Bài sau: Voice Activity Detection (phát hiện hoạt động tiếng nói) & Turn-Taking (luân phiên lượt nói) →](../../14-voice-activity-detection-turn-taking/docs/vi.md)
