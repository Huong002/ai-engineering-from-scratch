# Sinh nhạc (music generation) — MusicGen, Stable Audio, Suno, và cơn địa chấn bản quyền

> Sinh nhạc năm 2026: Suno v5 và Udio v4 thống trị mảng thương mại; MusicGen, Stable Audio Open và ACE-Step dẫn đầu mã nguồn mở. Bài toán kỹ thuật gần như đã xong. Bài toán pháp lý (Warner Music hoà giải 500 triệu USD, UMG hoà giải) đã định hình lại cả lĩnh vực trong 2025-2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Spectrogram), Giai đoạn 4, Bài 10 (Mô hình diffusion)
**Time:** ~75 phút

## Vấn đề

Văn bản → một clip nhạc dài 30 giây đến 4 phút, có lời, có giọng hát và có cấu trúc. Ba bài toán con:

1. **Sinh nhạc không lời.** Câu như "lo-fi hip-hop drums with warm keys" → âm thanh. MusicGen, Stable Audio, AudioLDM.
2. **Sinh bài hát (có giọng hát + lời).** "Country song about rainy Texas nights" → cả bài hát. Suno, Udio, YuE, ACE-Step.
3. **Có điều kiện / điều khiển được.** Kéo dài một clip có sẵn, sinh lại đoạn bridge, đổi thể loại, tách stem, hoặc inpaint. Inpainting + tách stem của Udio là tính năng năm 2026 mà ai cũng phải theo kịp.

## Khái niệm

![Sinh nhạc: token-LM so với khuếch tán, bản đồ mô hình 2026](../assets/music-generation.svg)

### Token LM trên các token neural-codec

**MusicGen** của Meta (2023, MIT) và rất nhiều biến thể: đặt conditioning (điều kiện hoá) trên embedding văn bản/giai điệu, dự đoán tự hồi quy (autoregressive) các token EnCodec (32 kHz, 4 codebook), rồi giải mã bằng EnCodec. 300M - 3.3B tham số. Baseline mạnh; hụt hơi khi vượt quá 30 giây.

**ACE-Step** (mã nguồn mở, bản 4B XL phát hành tháng 4/2026) mở rộng hướng này sang sinh cả bài hát có điều kiện theo lời. Đây là thứ gần Suno nhất mà cộng đồng mở có được.

### Khuếch tán trên mel hoặc latent

**Stable Audio (2023)** và **Stable Audio Open (2024)**: diffusion (khuếch tán) trên không gian latent (ẩn) của âm thanh đã nén. Rất mạnh ở loop, thiết kế âm thanh, texture ambient. Không giỏi ở các bài hát hoàn chỉnh có cấu trúc.

**AudioLDM / AudioLDM2**: văn bản thành âm thanh qua latent diffusion kiểu T2I, tổng quát hoá cho nhạc, hiệu ứng âm thanh, giọng nói.

### Lai (production) — Suno, Udio, Lyria

Trọng số đóng. Nhiều khả năng là AR codec LM + vocoder dựa trên diffusion với các head chuyên biệt cho giọng / trống / giai điệu. Suno v5 (2026) dẫn đầu chất lượng với ELO 1293. Udio v4 bổ sung inpainting + tách stem (tải riêng bass, trống, giọng).

### Đánh giá

- **FAD (Fréchet Audio Distance).** Khoảng cách ở mức embedding giữa phân phối âm thanh sinh ra và âm thanh thật, dùng đặc trưng VGGish hoặc PANNs. Càng thấp càng tốt. MusicGen small: FAD 4.5 trên MusicCaps; SOTA ~3.0.
- **Tính nhạc (chủ quan).** Sở thích của con người. Suno v5 ELO 1293 dẫn đầu.
- **Độ khớp văn bản-âm thanh.** Điểm CLAP giữa prompt và đầu ra.
- **Artifact về tính nhạc.** Chuyển đoạn lệch nhịp, trôi cụm từ hát, mất cấu trúc sau 30 giây.

## Bản đồ mô hình 2026

| Mô hình | Tham số | Độ dài | Giọng hát | Giấy phép |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | không | MIT |
| Stable Audio Open | 1.2B | 47 s | không | Stability phi thương mại |
| ACE-Step XL (4/2026) | 4B | &gt; 2 min | có | Apache-2.0 |
| YuE | 7B | &gt; 2 min | có, đa ngữ | Apache-2.0 |
| Suno v5 (đóng) | ? | 4 min | có, ELO 1293 | thương mại |
| Udio v4 (đóng) | ? | 4 min | có + stem | thương mại |
| Google Lyria 3 (đóng) | ? | thời gian thực | có | thương mại |
| MiniMax Music 2.5 | ? | 4 min | có | API thương mại |

## Bối cảnh pháp lý (2025-2026)

- **Warner Music vs Suno hoà giải.** 500 triệu USD. WMG giờ có quyền giám sát AI-likeness, quyền âm nhạc và các track do người dùng tạo trên Suno. Thoả thuận tương tự của UMG với Udio.
- **EU AI Act** + **California SB 942**: nhạc do AI tạo phải được công bố.
- **Riffusion / MusicGen** theo MIT không vướng gánh nặng tuân thủ, nhưng cũng không có giọng hát thương mại.

Các mẫu an toàn để đưa vào sản phẩm:

1. Chỉ sinh nhạc không lời (MusicGen, Stable Audio Open, đầu ra MIT/CC0).
2. Dùng API thương mại (Suno, Udio, ElevenLabs Music) với giấy phép theo từng lần sinh.
3. Huấn luyện trên catalog do bạn sở hữu hoặc đã cấp phép (phần lớn doanh nghiệp dừng ở đây).
4. Gắn watermark + metadata cho từng bản sinh.

```figure
sp-codec-tokens
```

## Tự tay làm

### Bước 1: sinh nhạc với MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Ba kích cỡ: `small` (300M, nhanh), `medium` (1.5B), `large` (3.3B). Bản small là đủ để trả lời câu "ý tưởng này có ổn không".

### Bước 2: điều kiện hoá theo giai điệu

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody nhận chromagram và giữ nguyên giai điệu trong khi đổi âm sắc. Hữu ích cho kiểu "cho tôi giai điệu này dưới dạng tứ tấu dây".

### Bước 3: đánh giá bằng FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Tính khoảng cách embedding VGGish. Hữu ích cho test hồi quy ở mức thể loại; không thay thế được người nghe.

### Bước 4: đưa vào quy trình LLM-music

Kết hợp với ý tưởng từ Bài 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Áp dụng

| Mục tiêu | Bộ công cụ |
|------|-------|
| Thiết kế âm thanh không lời | Stable Audio Open |
| Nhạc game / nhạc thích ứng | Google Lyria RealTime (đóng) |
| Bài hát đầy đủ có giọng (thương mại) | Suno v5 hoặc Udio v4 với giấy phép rõ ràng |
| Bài hát đầy đủ có giọng (mở) | ACE-Step XL hoặc YuE |
| Jingle quảng cáo ngắn | MusicGen điều kiện hoá theo giai điệu hum |
| Nền cho video nhạc | MusicGen + Stable Video Diffusion |

## Những cạm bẫy vẫn xuất hiện trong sản phẩm năm 2026

- **Prompt rửa bản quyền.** "Song in the style of Taylor Swift" — Suno/Udio thương mại đã lọc, mô hình mở thì chưa. Hãy tự thêm danh sách lọc của bạn.
- **Lặp / trôi sau 30 giây.** Mô hình AR bị loop. Crossfade nhiều bản sinh, hoặc dùng ACE-Step để có mạch cấu trúc.
- **Trôi tempo.** Mô hình lệch khỏi BPM. Gắn tag BPM trong prompt rồi lọc hậu bằng `beat_track` của librosa.
- **Độ rõ của giọng hát.** Suno rất tốt; mô hình mở thường nhoè chữ. Nếu lời quan trọng, dùng API thương mại hoặc fine-tune.
- **Đầu ra mono.** Mô hình mở sinh mono hoặc stereo giả. Nâng cấp bằng cách tái tạo stereo tử tế (ezst, stereo diffusion của Cartesia).

## Đóng gói

Lưu thành `outputs/skill-music-designer.md`. Chọn mô hình, chiến lược giấy phép, kế hoạch độ dài / cấu trúc, và metadata công bố cho một triển khai sinh nhạc.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Script tạo ra một vòng hợp âm (chord) "generative" + pattern trống dưới dạng ký hiệu ASCII — một bản nhạc cartoon. Phát lại bằng bất kỳ MIDI renderer nào nếu bạn muốn.
2. **Trung bình.** Cài `audiocraft`, sinh các clip 10 giây với 4 prompt thể loại khác nhau bằng MusicGen-small, đo FAD so với một bộ tham chiếu cùng thể loại.
3. **Khó.** Dùng ACE-Step (hoặc MusicGen-melody), sinh ba biến thể của cùng một giai điệu với các prompt âm sắc khác nhau. Tính độ tương đồng CLAP với prompt để kiểm chứng độ khớp.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| FAD | Audio FID | Khoảng cách Fréchet giữa phân phối embedding của âm thanh thật và âm thanh sinh ra. |
| Chromagram | Giai điệu dưới dạng cao độ | Vector 12 chiều cho mỗi frame; đầu vào của điều kiện hoá theo giai điệu. |
| Stems | Track nhạc cụ | Bass / trống / giọng / giai điệu đã tách riêng thành file WAV. |
| Inpainting | Sinh lại một đoạn | Che một cửa sổ thời gian; mô hình chỉ sinh lại đúng đoạn đó. |
| CLAP | CLIP cho văn bản-âm thanh | Embedding tương phản âm thanh-văn bản; dùng để đánh giá độ khớp văn bản-âm thanh. |
| EnCodec | Codec nhạc | Neural codec của Meta mà MusicGen dùng; 32 kHz, 4 codebook. |

## Đọc thêm

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — chuẩn mực tự hồi quy mở.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — lựa chọn mặc định cho thiết kế âm thanh.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — máy sinh cả bài hát 4B mã nguồn mở, tháng 4/2026.
- [Suno v5 platform docs](https://suno.com) — kẻ dẫn đầu chất lượng ở mảng thương mại.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latent diffusion cho nhạc + hiệu ứng âm thanh.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/warner-music-group-settles-with-suno-strikes-first-of-its-kind-deal-with-ai-song-generator/) — tiền lệ tháng 11/2025.

---

[← Bài trước: Sao chép giọng nói & Chuyển đổi giọng nói](../../08-voice-cloning-conversion/docs/vi.md) · [Bài sau: Mô hình ngôn ngữ âm thanh →](../../10-audio-language-models/docs/vi.md)
