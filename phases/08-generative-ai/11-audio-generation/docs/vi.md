# Sinh âm thanh

> Âm thanh là tín hiệu 1-D ở 16-48 kHz. Một clip năm giây là 80-240k mẫu. Không transformer nào attend trực tiếp vào chuỗi đó. Giải pháp cho mọi mô hình âm thanh production năm 2026 đều giống nhau: một neural codec (Encodec, SoundStream, DAC) nén âm thanh thành token rời rạc ở 50-75 Hz, rồi một transformer hoặc mô hình diffusion (khuếch tán) sinh ra token.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Đặc trưng âm thanh), Giai đoạn 6, Bài 04 (ASR), Giai đoạn 8, Bài 06 (DDPM)
**Time:** ~45 phút

## Vấn đề

Ba tác vụ audio generation (sinh âm thanh):

1. **Text-to-speech.** Cho văn bản, sinh giọng nói. Giọng nói sạch là băng hẹp và có cấu trúc ngữ âm mạnh — transformer-over-tokens giải tốt bài này. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Music generation.** Cho một prompt (văn bản, giai điệu, vòng hợp âm, thể loại), sinh nhạc. Phân phối rộng hơn nhiều. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Audio effects / sound design.** Cho một prompt, sinh âm thanh nền hoặc Foley. AudioGen, AudioLDM 2, Stable Audio Open.

Cả ba đều chạy trên cùng một nền: neural audio codec + token-AR hoặc bộ sinh diffusion.

## Khái niệm

![Sinh âm thanh: token codec + transformer hoặc diffusion](../assets/audio-generation.svg)

### Codec âm thanh thần kinh

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Một encoder tích chập nén waveform (dạng sóng) thành vector theo từng bước thời gian; lượng tử vector dư (RVQ) biến mỗi vector thành một chuỗi K chỉ số codebook. Decoder đảo ngược lại. Âm thanh 24 kHz ở 2 kbps dùng 8 codebook RVQ ở 75 Hz = 600 token/giây.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Hai mô hình sinh đặt lên trên

**Token-autoregressive.** Trải token RVQ thành một chuỗi, chạy một transformer chỉ có decoder. MusicGen dùng "delayed parallel" để phát ra K luồng codebook song song với độ lệch riêng từng luồng. VALL-E sinh token giọng nói từ prompt văn bản + mẫu giọng 3 giây.

**Khuếch tán ẩn (latent diffusion).** Đóng gói token codec thành latent liên tục hoặc mô hình hoá chúng bằng khuếch tán phân loại. Stable Audio 2.5 dùng flow matching trên latent âm thanh liên tục. AudioLDM 2 dùng khuếch tán văn bản-sang-mel-sang-âm thanh.

Xu hướng 2024-2026: flow matching đang thắng ở nhạc (suy luận nhanh hơn, mẫu sạch hơn) trong khi token-AR vẫn thống trị giọng nói vì nó tự nhiên có tính nhân quả và stream tốt.

## Bối cảnh production

| Hệ thống | Tác vụ | Backbone | Độ trễ |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + vocoder (bộ tổng hợp sóng) | ~300ms token đầu tiên |
| OpenAI GPT-4o audio | Giọng nói song công | AR đa phương thức đầu-cuối | ~200ms |
| NaturalSpeech 3 | TTS | Flow matching trên latent | Không stream |
| Stable Audio 2.5 | Nhạc / SFX | DiT + flow matching trên latent âm thanh | ~10s cho clip 1 phút |
| Suno v4 | Bài hát hoàn chỉnh | Không công bố; nghi là token-AR | ~30s mỗi bài |
| Udio v1.5 | Bài hát hoàn chỉnh | Không công bố | ~30s mỗi bài |
| MusicGen 3.3B | Nhạc | Token-AR trên Encodec 32kHz | Thời gian thực |
| AudioCraft 2 | Nhạc + SFX | Flow matching | ~5s cho clip 5s |
| Riffusion v2 | Nhạc | Khuếch tán spectrogram (phổ đồ) | ~10s |

```figure
score-matching
```

## Tự tay làm

`code/main.py` mô phỏng ý tưởng cốt lõi: huấn luyện một transformer dự đoán token kế tiếp nhỏ trên các chuỗi "audio token" tổng hợp sinh từ hai "phong cách" khác nhau (token thấp-cao xen kẽ cho phong cách A, ramp đơn điệu cho phong cách B). Điều kiện hoá theo phong cách rồi lấy mẫu.

### Bước 1: token âm thanh tổng hợp

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Bước 2: huấn luyện bộ dự đoán token nhỏ

Một bộ dự đoán kiểu bigram được điều kiện hoá theo phong cách. Ý là cái mẫu hình: token codec → huấn luyện cross-entropy → lấy mẫu tự hồi quy.

### Bước 3: lấy mẫu có điều kiện

Cho token phong cách và một token khởi đầu, lấy mẫu token kế tiếp từ phân phối dự đoán. Tiếp tục 20-40 token.

## Cạm bẫy

- **Chất lượng codec chặn trần chất lượng đầu ra.** Nếu codec không thể biểu diễn âm thanh trung thực (fidelity), generator có tốt đến đâu cũng không cứu được. DAC hiện là tốt nhất trong các bộ mở.
- **Lỗi tích luỹ của RVQ.** Mỗi lớp RVQ mô hình hoá phần dư của lớp trước. Lỗi ở lớp 1 lan truyền. Lấy mẫu với temperature 0 ở các lớp cao hơn sẽ giúp.
- **Cấu trúc âm nhạc.** 30 giây token là 20k+ token ở 75 Hz. Khó cho transformer. MusicGen dùng cửa sổ trượt + nối tiếp prompt; Stable Audio dùng clip ngắn hơn + crossfade.
- **Rác ở biên.** Crossfade giữa các clip được sinh cần overlap-add cẩn thận.
- **Thèm dữ liệu sạch.** Bộ sinh nhạc cần hàng chục nghìn giờ nhạc có giấy phép hợp lệ. Vụ kiện Suno / Udio của RIAA (2024) đưa chuyện này ra ánh sáng.
- **Đạo đức nhân bản giọng nói.** Một mẫu 3 giây cộng một prompt văn bản là đủ để VALL-E / XTTS / ElevenLabs nhân bản một giọng. Mọi mô hình production đều cần phát hiện lạm dụng + danh sách opt-out.

## Áp dụng

| Tác vụ | Stack 2026 |
|------|------------|
| TTS thương mại | ElevenLabs, OpenAI TTS, hoặc Azure Neural |
| Nhân bản giọng nói (đã xác minh đồng ý) | XTTS v2 (mở) hoặc ElevenLabs Pro |
| Nhạc nền, nhanh | Stable Audio 2.5 API, Suno, hoặc Udio |
| Nhạc có lời | Suno v4 hoặc Udio v1.5 |
| Hiệu ứng âm thanh / Foley | AudioCraft 2, ElevenLabs SFX, hoặc Stable Audio Open |
| Voice agent thời gian thực | GPT-4o realtime hoặc Gemini Live |
| Nghiên cứu nhạc open-weights | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Lồng tiếng / dịch | HeyGen, ElevenLabs Dubbing |

## Đóng gói

Lưu `outputs/skill-audio-brief.md`. Skill nhận một brief âm thanh (tác vụ, thời lượng, phong cách, giọng, giấy phép) và xuất ra: mô hình + hosting, định dạng prompt (tag thể loại, mô tả phong cách, dấu cấu trúc), chuỗi codec + generator + vocoder, giao thức seed, và kế hoạch eval (MOS / điểm CLAP / CER cho TTS / A/B người dùng).

## Bài tập

1. **Dễ.** Chạy `code/main.py` và đặt phong cách tường minh. Kiểm tra các chuỗi sinh ra khớp với mẫu của phong cách đó.
2. **Trung bình.** Thêm giải mã delayed parallel: mô phỏng 2 luồng token phải lệch nhau 1 bước. Huấn luyện một bộ dự đoán chung.
3. **Khó.** Dùng HuggingFace transformers để chạy MusicGen-small cục bộ. Sinh một clip 10 giây với ba prompt khác nhau; A/B xem mức bám phong cách.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Codec | "Nén thần kinh" | Encoder / decoder cho âm thanh; đầu ra điển hình là token 50-75 Hz. |
| RVQ | "VQ dư" | Chuỗi K bộ lượng tử; mỗi bộ mô hình hoá phần dư của bộ trước. |
| Token | "Một ký hiệu codec" | Chỉ số rời rạc trong một codebook; thường là 1024 hoặc 2048. |
| Delayed parallel | "Codebook lệch pha" | Phát ra K luồng token với độ lệch so le để giảm độ dài chuỗi. |
| Flow matching | "Cú thắng 2024 của âm thanh" | Phương án đường thẳng hơn thay cho diffusion; lấy mẫu nhanh hơn. |
| Voice prompt | "Mẫu 3 giây" | Embedding người nói hoặc tiền tố token dẫn hướng giọng được nhân bản. |
| Mel spectrogram | "Cái hình" | Phổ đồ log-magnitude theo cảm nhận; nhiều hệ TTS dùng nó. |
| Vocoder | "Mel ra sóng" | Thành phần neural chuyển phổ đồ mel trở lại âm thanh. |

## Ghi chú production: âm thanh là bài toán streaming

Âm thanh là modality đầu ra duy nhất mà người dùng mong nó đến *ngay khi đang được sinh*, không phải tất cả cùng lúc. Trong production, điều này nghĩa là TPOT quan trọng (Time Per Output Token) vì tốc độ nghe của người dùng là thông lượng mục tiêu — không phải tốc độ đọc. Với âm thanh 16kHz được token hoá ở ~75 token/giây (Encodec), server phải sinh ≥75 token/giây mỗi người dùng để phát lại mượt.

Hai hệ quả kiến trúc:

- **Mô hình âm thanh flow-matching không stream tầm thường được.** Stable Audio 2.5 và AudioCraft 2 render một clip có độ dài cố định trong một lượt. Để stream, bạn chia clip thành khối và chồng biên — kiểu khuếch tán cửa sổ trượt — thêm 100-300ms độ trễ so với mô hình codec AR.

Nếu sản phẩm là "voice chat trực tiếp" hoặc "nối tiếp nhạc thời gian thực", chọn đường codec AR. Nếu là "render một clip 30 giây khi bấm gửi", flow-matching thắng về chất lượng và tổng độ trễ.

## Đọc thêm

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — chuẩn codec.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — neural audio codec đầu tiên được dùng rộng rãi.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2025). Stable Audio 2.5](https://stability.ai/news-updates/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale) — text-to-music 2025 với flow matching.

---

[← Bài trước: Sinh video (Video Generation)](../../10-video-generation/docs/vi.md) · [Bài sau: Sinh 3D →](../../12-3d-generation/docs/vi.md)
