# Sao chép giọng nói & Chuyển đổi giọng nói

> Voice cloning (sao chép giọng nói) đọc văn bản của bạn bằng giọng của người khác. Voice conversion (chuyển đổi giọng nói) viết lại giọng của bạn thành giọng người khác mà vẫn giữ đúng điều bạn đã nói. Cả hai cùng dựa trên một cách phân rã: tách danh tính người nói khỏi nội dung.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 06 (Nhận dạng người nói), Giai đoạn 6, Bài 07 (TTS)
**Time:** ~75 phút

## Vấn đề

Năm 2026, một đoạn âm thanh 5 giây là đủ để tạo bản sao chất lượng cao giọng của bất kỳ ai bằng một GPU phổ thông. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox đều có sẵn tính năng sao chép zero-shot (không mẫu) hoặc few-shot. Công nghệ này là một phước lành (TTS trợ năng, lồng tiếng, giọng hỗ trợ) và cũng là một vũ khí (cuộc gọi lừa đảo, deepfake chính trị, đánh cắp sở hữu trí tuệ).

Hai tác vụ liên quan chặt chẽ:

- **Voice cloning (phía TTS):** văn bản + giọng tham chiếu 5 giây → âm thanh bằng giọng đó.
- **Voice conversion (phía giọng nói):** âm thanh nguồn (người A nói X) + giọng tham chiếu của người B → âm thanh của B nói X.

Cả hai đều phân rã một dạng sóng thành (nội dung, người nói, ngữ điệu) rồi ghép nội dung từ nguồn này với người nói từ nguồn kia.

Ràng buộc then chốt bạn phải chịu khi triển khai năm 2026: **gắn watermark (dấu thuỷ văn) và cổng kiểm tra consent (sự đồng ý) là yêu cầu pháp lý ở EU (AI Act, có hiệu lực tháng 8/2026) và ở California (AB 2905, có hiệu lực từ 2025)**. Pipeline của bạn phải phát ra một dấu thuỷ văn không nghe thấy được và từ chối các bản sao không có sự đồng ý.

## Khái niệm

![Sao chép giọng nói so với chuyển đổi: phân rã, hoán đổi người nói, ghép lại](../assets/voice-cloning.svg)

**Sao chép zero-shot.** Đưa một đoạn 5 giây vào một mô hình đã được huấn luyện trên hàng nghìn người nói. Bộ mã hoá người nói ánh xạ đoạn âm thanh thành speaker embedding (embedding người nói); bộ giải mã TTS điều kiện hoá trên embedding đó cộng với văn bản.

Được dùng bởi: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Fine-tune few-shot.** Ghi 5-30 phút giọng mục tiêu. Fine-tune LoRA một mô hình nền trong một giờ. Chất lượng nhảy vọt từ "tạm được" lên "không phân biệt được". Coqui và ElevenLabs đều hỗ trợ cách này; cộng đồng dùng nó với F5-TTS.

**Chuyển đổi giọng nói (VC).** Hai họ:

- **Nhận dạng-tổng hợp.** Chạy mô hình kiểu ASR để trích biểu diễn nội dung (ví dụ: xác suất phoneme mềm, PPG), rồi tổng hợp lại với embedding người nói đích. Bền với ngôn ngữ và giọng địa phương. Được dùng bởi KNN-VC (2023), Diff-HierVC (2023).
- **Tách rời (disentanglement).** Huấn luyện một autoencoder tách nội dung, người nói và ngữ điệu trong không gian latent tại nút thắt cổ chai. Hoán đổi embedding người nói khi suy luận. Chất lượng thấp hơn nhưng nhanh hơn. Được dùng bởi AutoVC (2019), các biến thể VITS-VC.

**Sao chép dựa trên neural codec (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — coi âm thanh là các token rời rạc từ SoundStream / EnCodec, huấn luyện một mô hình autoregressive hoặc flow-matching lớn trên các token codec. Chất lượng tương đương ElevenLabs trên prompt ngắn.

### Phần đạo đức, không phải thứ gắn thêm

**Gắn dấu thuỷ văn.** PerTh (Perth) và SilentCipher (2024) nhúng một ID ~16-32 bit vào âm thanh mà tai không nghe ra. Sống sót qua mã hoá lại, streaming và các chỉnh sửa thông thường. Mã nguồn mở, sẵn sàng cho production.

**Cổng kiểm tra sự đồng ý.** Mỗi đầu ra sao chép phải đi kèm một bản ghi đồng ý có thể xác minh. "Tôi, Rohit, ngày 2026-04-22, cho phép giọng này cho mục đích X." Lưu trong một sổ ghi chống giả mạo.

**Phát hiện.** AASIST, RawNet2 và Wav2Vec2-AASIST có sẵn như các bộ phát hiện. Thử thách ASVspoof 2025 công bố EER 0.8–2.3% cho các bộ phát hiện hiện đại nhất đối với đầu ra của ElevenLabs, VALL-E 2 và Bark.

### Số liệu (2026)

| Mô hình | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Có | 0.72 | 2.1% | 335M |
| XTTS v2 | Có | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Có | 0.70 | 2.8% | 220M |
| VALL-E 2 | Có | 0.77 | 2.4% | 370M |
| VoiceBox | Có | 0.78 | 2.1% | 330M |

SECS > 0.70 thường được hầu hết người nghe coi là không phân biệt được với giọng mục tiêu.

```figure
sp-voice-factorize
```

## Tự tay làm

### Bước 1: phân rã bằng nhận dạng-tổng hợp (bản demo chỉ-code trong main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Về khái niệm thì đơn giản; phần nặng của triển khai nằm trong `tts_model` và bộ mã hoá người nói.

### Bước 2: sao chép zero-shot với F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Bản ghi lời tham chiếu phải khớp chính xác với âm thanh; lệch sẽ phá vỡ căn chỉnh.

### Bước 3: chuyển đổi giọng nói với KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC chạy WavLM để trích embedding theo từng frame cho nguồn và tập đích, rồi thay mỗi frame nguồn bằng láng giềng gần nhất của nó trong tập. Không tham số, hoạt động với một phút giọng mục tiêu.

### Bước 4: nhúng dấu thuỷ văn

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bit payload, phát hiện được sau khi mã hoá lại MP3 và nhiễu nhẹ.

### Bước 5: cổng kiểm tra sự đồng ý

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Áp dụng

Bộ công cụ 2026:

| Tình huống | Chọn |
|-----------|------|
| Sao chép zero-shot 5 giây, mã nguồn mở | F5-TTS hoặc OpenVoice v2 |
| Sao chép thương mại cho production | ElevenLabs Instant Voice Clone v2.5 |
| Chuyển đổi giọng nói (viết lại) | KNN-VC hoặc Diff-HierVC |
| Fine-tune nhiều người nói | StyleTTS 2 + speaker adapter |
| Sao chép xuyên ngôn ngữ | XTTS v2 hoặc VALL-E X |
| Phát hiện deepfake | Wav2Vec2-AASIST |

## Cạm bẫy

- **Bản ghi lời tham chiếu lệch.** F5-TTS và các mô hình tương tự yêu cầu văn bản tham chiếu khớp chính xác với âm thanh tham chiếu, kể cả dấu câu.
- **Giọng tham chiếu có vang.** Tiếng vọng giết bản sao. Hãy ghi khô, mic gần.
- **Lệch cảm xúc.** Tham chiếu huấn luyện "vui vẻ" sẽ tạo ra bản sao vui vẻ cho mọi thứ. Khớp cảm xúc tham chiếu với mục đích sử dụng.
- **Rò rỉ ngôn ngữ.** Sao chép một người nói tiếng Anh rồi bảo mô hình nói tiếng Pháp thường vẫn mang theo giọng đó; hãy dùng mô hình xuyên ngôn ngữ (XTTS, VALL-E X).
- **Không có dấu thuỷ văn.** Không thể triển khai hợp pháp ở EU từ tháng 8/2026.

## Đóng gói

Lưu thành `outputs/skill-voice-cloner.md`. Thiết kế một pipeline sao chép hoặc chuyển đổi với cổng kiểm tra sự đồng ý + dấu thuỷ văn + mục tiêu chất lượng.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Minh hoạ việc hoán đổi speaker embedding bằng cách tính cosine giữa hai "người nói" trước và sau khi hoán đổi.
2. **Trung bình.** Dùng OpenVoice v2 để sao chép giọng của chính bạn. Đo SECS giữa tham chiếu và bản sao. Đo CER bằng Whisper.
3. **Khó.** Áp dấu thuỷ văn SilentCipher lên 20 bản sao, đưa chúng qua mã hoá+giải mã MP3 128 kbps, phát hiện payload. Báo cáo độ chính xác bit.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 giây là đủ | Mô hình pretrained + speaker embedding; không huấn luyện. |
| PPG | Phonetic posteriorgram | Xác suất ASR theo từng frame dùng làm biểu diễn nội dung không phụ thuộc ngôn ngữ. |
| KNN-VC | Chuyển đổi bằng láng giềng gần nhất | Thay mỗi frame nguồn bằng frame gần nhất trong tập đích. |
| Neural codec TTS | Kiểu VALL-E | Mô hình AR trên các token EnCodec/SoundStream. |
| Watermark | Chữ ký không nghe thấy | Các bit được nhúng trong âm thanh, sống sót qua mã hoá lại. |
| SECS | Độ trung thực bản sao | Cosine giữa embedding người nói đích và bản sao. |
| AASIST | Bộ phát hiện deepfake | Mô hình chống giả mạo; phát hiện giọng tổng hợp. |

## Đọc thêm

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — sao chép zero-shot SOTA mã nguồn mở.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) và [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — TTS neural-codec.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — chuyển đổi giọng nói dựa trên tách rời.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — VC dựa trên truy xuất.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — dấu thuỷ văn âm thanh 32-bit sẵn sàng production.
- [Kết quả ASVspoof 2025](https://www.asvspoof.org/) — cuộc đua vũ trang giữa bộ phát hiện và bộ tổng hợp, cập nhật 2026.

---

[← Bài trước: Text-to-Speech (chuyển văn bản thành giọng nói)](../../07-text-to-speech/docs/vi.md) · [Bài sau: Sinh nhạc (music generation) →](../../09-music-generation/docs/vi.md)
