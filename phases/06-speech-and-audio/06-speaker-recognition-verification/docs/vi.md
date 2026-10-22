# Nhận dạng & Xác minh người nói

> ASR hỏi "họ đã nói gì?" Nhận dạng người nói hỏi "ai đã nói điều đó?" Phép toán trông giống nhau — embedding cộng cosine — nhưng mọi quyết định triển khai đều xoay quanh một con số EER duy nhất.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Spectrogram & Mel), Giai đoạn 5, Bài 22 (Mô hình embedding)
**Time:** ~45 phút

## Vấn đề

Một người dùng nói một câu mật khẩu. Bạn muốn biết: đây có đúng là người mà họ tự nhận không (*speaker verification* (xác minh người nói), 1:1), hay đây là người đầu tiên trong kho enrollment (đăng ký mẫu giọng) của bạn (*speaker identification* (nhận dạng người nói), 1:N)? Hay không phải cả hai — đây là một người nói lạ (*open-set*)?

Trước 2018: GMM-UBM + i-vectors. EER khá ổn nhưng dễ vỡ khi kênh thay đổi (điện thoại so với laptop) và khi có cảm xúc. 2018–2022: x-vector (backbone TDNN huấn luyện với angular margin). 2022 trở đi: embedding ECAPA-TDNN và WavLM-large. Đến 2026, lĩnh vực này bị chi phối bởi ba mô hình và một thước đo.

Thước đo đó là **EER** — equal error rate (tỉ lệ lỗi cân bằng). Bạn đặt ngưỡng quyết định sao cho tỉ lệ chấp nhận sai bằng tỉ lệ từ chối sai. Điểm giao nhau đó chính là EER. Nó có mặt trong mọi bài báo, mọi bảng xếp hạng, mọi cuộc đàm phán mua thiết bị.

## Khái niệm

![Pipeline enrollment + verification với embedding + cosine + EER](../assets/speaker-verification.svg)

**Đường ống xử lý.** Enrollment: ghi 5–30 giây giọng của người nói mục tiêu; tính một embedding có số chiều cố định (192 chiều với ECAPA-TDNN, 256 chiều với WavLM-large). Verification: lấy embedding của câu nói kiểm thử; tính cosine similarity; so với một ngưỡng.

**ECAPA-TDNN (2020, vẫn chi phối năm 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. Các khối conv 1D với squeeze-excitation, pooling bằng multi-head attention, rồi một lớp linear về 192 chiều. Huấn luyện trên VoxCeleb 1+2 (2.700 người nói, 1,1M câu nói) với Additive Angular Margin loss (AAM-softmax).

**WavLM-SV (2022 trở đi).** Fine-tune một backbone SSL WavLM-large đã huấn luyện trước bằng AAM loss. Chất lượng cao hơn nhưng chậm hơn — 300+ MB so với 15 MB.

**x-vector (đường cơ sở).** TDNN + statistics pooling. Kinh điển; vẫn hữu ích trên CPU / edge.

**AAM-softmax.** Softmax chuẩn cộng thêm margin `m` trong không gian góc: `cos(θ + m)` cho lớp đúng. Buộc các lớp tách xa nhau về góc. Giá trị thường dùng `m=0.2`, scale `s=30`.

### Tính điểm

- **Cosine** giữa embedding enrollment và embedding kiểm thử. Quyết định dựa trên ngưỡng.
- **PLDA (Probabilistic LDA).** Chiếu embedding vào một không gian ẩn, nơi cùng-người-nói so với khác-người-nói có tỉ số hợp lý dạng đóng. Thêm lên trên cosine để giảm EER 10–20%. Là chuẩn trước 2020; nay chỉ còn dùng trong bối cảnh closed-set.
- **Score normalization** (chuẩn hoá điểm). `S-norm` hoặc `AS-norm`: chuẩn hoá từng điểm theo trung bình và độ lệch chuẩn của một nhóm impostor (kẻ mạo danh). Thiết yếu khi đánh giá khác miền.

### Những con số bạn nên biết (2026)

| Mô hình | EER trên VoxCeleb1-O | Tham số | Thông lượng (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (kinh điển) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 phân đoạn + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization

"Ai nói khi nào" trong một clip nhiều người nói. Pipeline: VAD → phân đoạn → embed từng đoạn → cluster (agglomerative hoặc spectral) → làm mượt biên. Stack hiện đại: `pyannote.audio` 3.1, gói sẵn phân đoạn người nói + embedding + clustering sau một lời gọi. SOTA DER trên AMI năm 2026 là ~15% (giảm từ 23% năm 2022).

```figure
sp-eer-crossover
```

## Tự tay làm

### Bước 1: embedding đồ chơi từ thống kê MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Còn xa mới đạt SOTA — chỉ để dạy học. `code/main.py` dùng cách này như một bản thử nghiệm trên dữ liệu người nói tổng hợp.

### Bước 2: cosine similarity + ngưỡng

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Bước 3: EER từ các cặp điểm tương đồng

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Trả về (eer, threshold_at_eer). Hãy báo cáo cả hai.

### Bước 4: triển khai với SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Bước 5: tách người nói với pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Áp dụng

Stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Xác minh 1:1 closed-set, trên thiết bị biên | ECAPA-TDNN + ngưỡng cosine |
| Xác minh open-set, trên cloud | WavLM-SV + AS-norm |
| Tách người nói (cuộc họp, podcast) | `pyannote/speaker-diarization-3.1` |
| Chống giả mạo (phát hiện replay / deepfake) | AASIST hoặc RawNet2 |
| Thiết bị nhúng nhỏ (KWS + enrollment) | Titanet-Small (NeMo) |

## Cạm bẫy

- **Lệch kênh.** Mô hình huấn luyện trên VoxCeleb (video web) ≠ âm thanh cuộc gọi điện thoại. Luôn đánh giá trên đúng kênh mục tiêu.
- **Câu nói ngắn.** EER xấu đi rõ rệt khi âm thanh kiểm thử dưới 3 giây.
- **Enrollment nhiễu.** Một mẫu enrollment nhiễu làm hỏng mốc neo. Dùng ≥3 mẫu sạch rồi lấy trung bình.
- **Dùng một ngưỡng cố định cho mọi điều kiện.** Luôn tinh chỉnh ngưỡng trên một tập dev giữ riêng lấy từ miền mục tiêu.
- **Cosine trên embedding chưa chuẩn hoá.** Hãy chuẩn hoá L2 trước; nếu không, độ lớn sẽ chi phối.

## Đóng gói

Lưu thành `outputs/skill-speaker-verifier.md`. Chọn mô hình, quy trình enrollment, kế hoạch tinh chỉnh ngưỡng và các biện pháp chống gian lận.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó dựng các "speaker" tổng hợp (các profile âm sắc khác nhau), đăng ký mẫu, rồi tính EER trên danh sách 100 cặp trial (lần thử).
2. **Trung bình.** Dùng SpeechBrain ECAPA trên 30 câu nói VoxCeleb1 (5 người nói × 6 câu mỗi người). Tính EER với cosine so với PLDA.
3. **Khó.** Dựng toàn bộ pipeline enroll → diarize → verify với `pyannote.audio`. Đánh giá DER trên tập dev AMI.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| EER | Chỉ số hàng đầu | Ngưỡng mà Chấp nhận sai = Từ chối sai. |
| Verification | 1:1 | "Đây có phải Alice không?" |
| Identification | 1:N | "Ai đang nói?" |
| Open-set | Có thể có người lạ | Tập kiểm thử có thể chứa người nói chưa đăng ký. |
| Enrollment | Đăng ký | Tính embedding tham chiếu cho một người nói. |
| AAM-softmax | Hàm loss | Softmax với margin góc cộng thêm; buộc các cụm tách xa nhau. |
| PLDA | Cách tính điểm kinh điển | Probabilistic LDA; tính điểm bằng tỉ số hợp lý trên nền embedding. |
| DER | Thước đo tách người nói | Diarization Error Rate — bỏ sót + báo nhầm + nhầm lẫn. |

## Đọc thêm

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — bài báo kinh điển về deep embedding.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — kiến trúc chi phối giai đoạn 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — backbone SSL cho SV và tách người nói.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — stack tách người nói + embedding dùng trong production.
- [Bảng xếp hạng VoxCeleb (cập nhật 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — thứ hạng EER hiện hành của các mô hình.

---

[← Bài trước: Whisper](../../05-whisper-architecture-finetuning/docs/vi.md) · [Bài sau: Text-to-Speech (chuyển văn bản thành giọng nói) →](../../07-text-to-speech/docs/vi.md)
