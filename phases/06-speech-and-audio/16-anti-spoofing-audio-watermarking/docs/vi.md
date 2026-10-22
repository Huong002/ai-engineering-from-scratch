# Chống giả mạo giọng nói & dấu thuỷ văn âm thanh — ASVspoof 5, AudioSeal, WaveVerify

> Sao chép giọng nói ra mắt nhanh hơn các biện pháp phòng thủ. Hệ thống giọng nói production năm 2026 cần hai thứ: một bộ phát hiện (AASIST, RawNet2) phân loại giọng thật và giọng giả, cùng một dấu thuỷ văn (watermark) sống sót qua nén và chỉnh sửa. Hãy đóng gói cả hai, hoặc đừng đóng gói sao chép giọng nói.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 06 (Nhận dạng người nói), Giai đoạn 6, Bài 08 (Sao chép giọng nói)
**Time:** ~75 phút

## Vấn đề

Ba lớp phòng thủ liên quan:

1. **Chống giả mạo (anti-spoofing) / phát hiện (detection) deepfake.** Cho một clip âm thanh, nó là tổng hợp hay thật? Các bộ đánh giá ASVspoof (ASVspoof 2019 → 2021 → 5) là chuẩn vàng.
2. **Dấu thuỷ văn âm thanh.** Nhúng một tín hiệu không nghe thấy vào âm thanh được sinh ra để bộ phát hiện có thể trích xuất về sau. AudioSeal (Meta) và WavMark là các lựa chọn mở.
3. **Nguồn gốc (provenance) có xác thực.** Ký số tệp âm thanh + metadata. C2PA / Content Authenticity Initiative.

Phát hiện xử lý những kẻ tấn công không hợp tác. Dấu thuỷ văn xử lý việc tuân thủ — âm thanh do AI tạo nên có thể nhận ra được là như vậy. Năm 2026 cần cả hai.

## Khái niệm

![Chống giả mạo so với dấu thuỷ văn so với nguồn gốc — ba lớp phòng thủ](../assets/spoofing-watermark.svg)

### ASVspoof 5 — bộ đánh giá 2024-2025

Thay đổi lớn nhất so với các phiên bản trước:

- **Dữ liệu quần chúng** (không phải studio sạch) — điều kiện thực tế.
- **~2000 người nói** (so với ~100 trước đây).
- **32 thuật toán tấn công.** TTS + chuyển đổi giọng + nhiễu đối kháng.
- **Hai track.** Countermeasure (CM) phát hiện độc lập; Spoofing-robust ASV (SASV) cho hệ thống sinh trắc học.

Hiện trạng tốt nhất trên ASVspoof 5: ~7.23% EER. Trên ASVspoof 2019 LA cũ hơn: 0.42% EER. Triển khai thực tế: hãy dự trù 5-10% EER trên các clip in-the-wild.

### AASIST và RawNet2 — các họ mô hình phát hiện

**AASIST** (2021, cập nhật đến 2026). Graph-attention trên các đặc trưng phổ. SOTA hiện tại trên tác vụ countermeasure của ASVspoof 5.

**RawNet2.** Front-end tích chập trên waveform thô + xương sống TDNN. Baseline đơn giản hơn; vẫn cạnh tranh khi fine-tuning.

**NeXt-TDNN + đặc trưng SSL.** Biến thể 2025: kiểu ECAPA + đặc trưng WavLM + focal loss. Đạt EER 0.42% trên ASVspoof 2019 LA.

### AudioSeal — mặc định watermark 2024

**AudioSeal** của Meta (tháng 1/2024, v0.2 tháng 12/2024). Thiết kế chính:

- **Cục bộ.** Phát hiện dấu thuỷ văn theo từng frame ở độ phân giải mẫu 16 kHz (1/16000 s).
- **Generator + detector huấn luyện chung.** Generator học cách nhúng tín hiệu không nghe được; detector học cách tìm ra nó qua các phép tăng cường.
- **Bền.** Sống sót qua nén MP3 / AAC, EQ, dịch tốc độ ±10%, trộn nhiễu +10 dB SNR.
- **Nhanh.** Detector chạy ở 485× thời gian thực; nhanh hơn WavMark 1000×.
- **Dung lượng.** Payload 16 bit (có thể mã hoá model ID, mốc thời gian sinh, user ID) nhúng được trong từng utterance.

### WavMark

Baseline mở trước AudioSeal. Mạng nơ-ron khả nghịch, 32 bit/giây. Vấn đề:

- Đồng bộ bằng brute-force thì chậm.
- Có thể bị xoá bằng nhiễu Gaussian hoặc nén MP3.
- Không thân thiện với thời gian thực.

### WaveVerify (tháng 7/2025)

Khắc phục điểm yếu của AudioSeal — cụ thể là các thao tác thời gian (đảo ngược, tốc độ). Dùng generator dựa trên FiLM + detector Mixture-of-Experts. Cạnh tranh với AudioSeal trên các tấn công chuẩn; xử lý được chỉnh sửa thời gian.

### Khoảng trống mà kẻ tấn công khai thác

Từ AudioMarkBench: "dưới dịch cao độ, mọi dấu thuỷ văn đều cho Bit Recovery Accuracy dưới 0.6, cho thấy gần như bị xoá hoàn toàn." **Dịch cao độ là đòn tấn công phổ quát.** Không dấu thuỷ văn nào năm 2026 bền hoàn toàn với thay đổi cao độ mạnh. Đây là lý do bạn cần phát hiện (AASIST) song song với dấu thuỷ văn.

### C2PA / Content Authenticity Initiative

Không phải một kỹ thuật ML — mà là một định dạng manifest. Tệp âm thanh mang metadata được ký bằng mật mã về công cụ tạo, tác giả, ngày. Audobox / Seamless dùng nó. Tốt cho nguồn gốc; không giúp gì nếu kẻ xấu mã hoá lại và bóc metadata.

```figure
v4-audio-watermark
```

## Tự tay làm

### Bước 1: một bộ phát hiện đặc trưng phổ đơn giản (đồ chơi)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Giọng tổng hợp thường có năng lượng tần số cao phẳng một cách bất thường. Detector production dùng AASIST, không dùng cái này. Nhưng trực giác vẫn đúng.

### Bước 2: AudioSeal nhúng + phát hiện

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Bước 3: đánh giá — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Bước 4: tích hợp production

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Mỗi lần sinh ra đều giao kèm: (1) dấu thuỷ văn, (2) manifest đã ký, (3) audit log tuân thủ chính sách lưu trữ.

## Áp dụng

| Trường hợp sử dụng | Phòng thủ |
|----------|---------|
| Đóng gói TTS / sao chép giọng nói | Nhúng AudioSeal trên mọi đầu ra (không thể thương lượng) |
| Mở khoá bằng giọng sinh trắc học | Ensemble AASIST + ECAPA; thử thách liveness |
| Phát hiện gian lận ở call-center | AASIST trên 20% mẫu cuộc gọi đến |
| Xác thực podcast | Ký C2PA khi tải lên, AudioSeal nếu do AI tạo |
| Nghiên cứu / huấn luyện detector | Bộ train/dev/eval của ASVspoof 5 |

## Cạm bẫy

- **Dấu thuỷ văn mà detector không bao giờ chạy.** Vô nghĩa. Hãy đóng gói detector vào CI của bạn.
- **Phát hiện mà không hiệu chỉnh.** AASIST huấn luyện trên ASVspoof LA bị overfit; độ chính xác thực tế giảm. Hãy hiệu chỉnh trên miền dữ liệu của bạn.
- **Khoảng trống dịch cao độ.** Dịch cao độ mạnh xoá hầu hết dấu thuỷ văn. Phải có phương án phát hiện dự phòng.
- **Bóc metadata rồi đăng lại.** C2PA bị vượt qua dễ dàng bằng cách mã hoá lại. Luôn thêm phòng thủ mật mã + tri giác (dấu thuỷ văn) cùng nhau.
- **Liveness như một cách phát hiện.** Yêu cầu người dùng nói một cụm từ ngẫu nhiên. Ngăn được tấn công phát lại nhưng không ngăn được sao chép thời gian thực.

## Đóng gói

Lưu thành `outputs/skill-spoof-defender.md`. Chọn mô hình phát hiện, dấu thuỷ văn, manifest nguồn gốc và playbook vận hành cho một triển khai voice-gen.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Detector đồ chơi + nhúng/phát hiện dấu thuỷ văn đồ chơi trên âm thanh tổng hợp.
2. **Trung bình.** Cài `audioseal`, nhúng một payload 16 bit vào đầu ra TTS, giải mã lại. Làm hỏng âm thanh bằng nhiễu và đo Bit Recovery Accuracy.
3. **Khó.** Fine-tune một RawNet2 hoặc AASIST trên ASVspoof 2019 LA. Đo EER. Kiểm thử trên một tập giữ riêng gồm các clip do F5-TTS sinh ra — xem phát hiện OOD suy giảm ra sao.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| ASVspoof | Bộ đánh giá | Cuộc thi hai năm một lần; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detector | Bộ phân loại: giọng thật so với giọng tổng hợp / đã chuyển đổi. |
| SASV | Xác minh người nói + CM | Tích hợp sinh trắc học + phát hiện giả mạo. |
| AudioSeal | Watermark của Meta | Cục bộ, payload 16 bit, nhanh hơn WavMark 485×. |
| Bit Recovery Accuracy | Độ sống sót của watermark | Tỉ lệ bit payload thu hồi được sau tấn công. |
| C2PA | Manifest nguồn gốc | Metadata mật mã về việc tạo / tác giả. |
| AASIST | Họ mô hình phát hiện | Anti-spoofing SOTA dựa trên graph-attention. |

## Đọc thêm

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — bộ đánh giá hiện tại.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — mặc định watermark.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — detector MoE cho tấn công thời gian.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — xương sống phát hiện SOTA.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — đánh giá độ bền.
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/index.html) — định dạng manifest nguồn gốc.

---

[← Bài trước: Streaming speech-to-speech](../../15-streaming-speech-to-speech-moshi-hibiki/docs/vi.md) · [Bài sau: Đánh giá âm thanh →](../../17-audio-evaluation-metrics/docs/vi.md)
