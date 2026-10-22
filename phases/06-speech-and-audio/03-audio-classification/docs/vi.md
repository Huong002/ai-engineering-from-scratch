# Phân loại âm thanh — Từ k-NN trên MFCC tới AST và BEATs

> Mọi thứ từ "chó sủa vs còi báo động" đến "đây là ngôn ngữ nào" đều là audio classification (phân loại âm thanh). Đặc trưng là mel. Kiến trúc thay đổi theo từng thập kỷ. Còn đánh giá vẫn là AUC, F1, và recall theo từng lớp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Spectrogram & Mel), Giai đoạn 3, Bài 06 (CNN), Giai đoạn 5, Bài 08 (CNN & RNN cho văn bản)
**Time:** ~75 phút

## Vấn đề

Bạn có một đoạn âm thanh 10 giây. Bạn muốn biết: "đó là gì?" Âm thanh đô thị (còi báo động, máy khoan, chó sủa), lệnh thoại (yes/no/stop), nhận dạng ngôn ngữ (en/es/ar), cảm xúc người nói (giận/trung tính), hay âm thanh môi trường (trong nhà/ngoài trời, tiếng nói ríu rít). Tất cả đều là *audio classification*, và tới 2026 kiến trúc baseline đã chín muồi: log-mel → CNN hoặc Transformer → softmax.

Khó khăn cốt lõi không nằm ở mạng nơ-ron. Mà ở dữ liệu. Bộ dữ liệu âm thanh có mất cân bằng lớp (class imbalance) cực nặng, lệch miền mạnh (sạch vs nhiễu), và nhãn nhiễu (ai quyết định "tiếng nói ríu rít đô thị" vs "tiếng ồn nhà hàng"?). 80% vấn đề là tuyển chọn dữ liệu, tăng cường dữ liệu (data augmentation), và đánh giá, chứ không phải đổi CNN sang Transformer.

## Khái niệm

![Thang bậc phân loại âm thanh: k-NN trên MFCC tới AST tới BEATs](../assets/audio-classification.svg)

**k-NN trên MFCC (baseline thập niên 1990).** Trải phẳng MFCC của từng clip, tính độ tương đồng cosine với một ngân hàng mẫu đã gán nhãn, trả về bầu chọn đa số của K láng giềng gần nhất. Mạnh bất ngờ trên bộ dữ liệu nhỏ và sạch (Speech Commands, ESC-50). Chạy không cần GPU.

**CNN 2D trên log-mel (2015-2019).** Coi log-mel `(T, n_mels)` như một bức ảnh. Áp ResNet-18 hay kiểu VGG. Pool (gộp) trung bình toàn cục theo trục thời gian. Softmax trên các lớp. Vẫn là baseline ở phần lớn cuộc thi kaggle 2026.

**Audio Spectrogram Transformer, AST (2021-2024).** Chia log-mel thành các patch (ví dụ patch 16×16), thêm position embedding, đưa vào ViT. State of the art trên AudioSet (mAP 0.485) cho học có giám sát.

**BEATs và WavLM-base (2024-2026).** Tiền huấn luyện tự giám sát trên hàng triệu giờ âm thanh. Fine-tune cho tác vụ của bạn với 1-10% dữ liệu có giám sát mà lẽ ra bạn cần. Tới 2026 đây là điểm khởi đầu mặc định cho âm thanh phi thoại. BEATs-iter3 vượt AST 1-2 mAP trên AudioSet mà chỉ tốn 1/4 compute.

**Whisper-encoder làm backbone đóng băng (2024).** Lấy encoder của Whisper, bỏ decoder, gắn một bộ phân loại tuyến tính. Gần SOTA cho nhận dạng ngôn ngữ và phân loại sự kiện đơn giản mà không cần tăng cường dữ liệu âm thanh. Baseline "bữa trưa miễn phí".

### Mất cân bằng lớp mới là thách thức thật sự

ESC-50: 50 lớp, 40 clip mỗi lớp — cân bằng, dễ. UrbanSound8K: 10 lớp, mất cân bằng tỉ lệ 10:1. AudioSet: 632 lớp với đuôi dài 100.000:1. Những kỹ thuật hiệu quả:

- Lấy mẫu cân bằng trong lúc huấn luyện (không dùng khi đánh giá).
- Mixup: nội suy tuyến tính hai clip (và nhãn của chúng) như một phép tăng cường dữ liệu.
- SpecAugment: che các dải thời gian và tần số ngẫu nhiên. Đơn giản; nhưng quan trọng.

### Đánh giá

- Đa lớp loại trừ (Speech Commands): top-1 accuracy (độ chính xác top-1), top-5 accuracy.
- Đa lớp đa nhãn (AudioSet, kiểu UrbanSound): mean average precision (mAP).
- Mất cân bằng nặng: recall theo từng lớp + macro F1.

Các con số 2026 bạn nên biết:

| Benchmark | Baseline | SOTA 2026 | Nguồn |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | Bài báo BEATs (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | Bảng xếp hạng HEAR 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | Kết quả HEAR v2 |

```figure
mfcc-pipeline
```

## Tự tay làm

### Bước 1: featurize

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Bước 2: tóm tắt về độ dài cố định

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Đơn giản nhưng mạnh: mean + variance theo thời gian cho một embedding cố định 26 chiều từ MFCC 13 hệ số. Chạy tức thì. Từng vượt các baseline mạng nơ-ron state-of-the-art trên ESC-50 vào tận năm 2017.

### Bước 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Bước 4: nâng cấp lên CNN trên log-mel

Trong PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M tham số. Huấn luyện trong ~10 phút trên ESC-50 với một chiếc RTX 4090. Độ chính xác 80%+.

### Bước 5: fine-tune một audio transformer đã tiền huấn luyện (ví dụ AST)

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Ví dụ trên fine-tune AST từ Hub. BEATs, mặc định của 2026, không có trên Hugging Face Hub: hãy tải checkpoint từ [BEATs release in microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats) và load bằng các lớp `BEATs` và `BEATsConfig` của repo đó; vòng fine-tune giữ nguyên hình dạng như vậy.

## Áp dụng

Bộ công cụ 2026:

| Tình huống | Bắt đầu với |
|-----------|-----------|
| Bộ dữ liệu tí xíu (<1000 clip) | k-NN trên trung bình MFCC (baseline của bạn) + tăng cường dữ liệu âm thanh |
| Bộ dữ liệu vừa (1K–100K) | Fine-tune BEATs hoặc AST |
| Bộ dữ liệu lớn (>100K) | Huấn luyện từ đầu hoặc fine-tune Whisper-encoder |
| Thời gian thực, thiết bị biên | CNN 40-MFCC, lượng tử hoá về int8 (kiểu KWS) |
| Đa nhãn (AudioSet) | BEATs-iter3 với loss BCE + mixup + SpecAugment |
| Nhận dạng ngôn ngữ | MMS-LID, baseline SpeechBrain VoxLingua107 |

Quy tắc quyết định: **hãy bắt đầu với một backbone đóng băng, không phải một mô hình mới**. Fine-tune một head BEATs cho bạn 95% SOTA trong vài giờ, không phải vài tuần.

## Đóng gói

Lưu thành `outputs/skill-classifier-designer.md`. Chọn kiến trúc, các phép tăng cường, chiến lược cân bằng lớp, và metric đánh giá cho một tác vụ phân loại âm thanh cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó huấn luyện baseline k-NN MFCC trên một bộ dữ liệu tổng hợp 4 lớp (âm thuần với cao độ khác nhau). Báo cáo confusion matrix (ma trận nhầm lẫn).
2. **Trung bình.** Thay `summarize` bằng [mean, var, skew, kurtosis]. Pooling (gộp) 4 moment có vượt mean+var trên cùng bộ dữ liệu tổng hợp đó không?
3. **Khó.** Dùng `torchaudio`, huấn luyện một CNN 2D trên fold 1 của ESC-50. Báo cáo độ chính xác cross-validation 5 fold. Thêm SpecAugment (time mask = 20, freq mask = 10) và báo cáo mức chênh lệch.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| AudioSet | ImageNet của âm thanh | Bộ dữ liệu YouTube 2 triệu clip, 632 lớp, gán nhãn yếu của Google. |
| ESC-50 | Bộ benchmark phân loại nhỏ | 50 lớp × 40 clip âm thanh môi trường. |
| AST | Audio Spectrogram Transformer | ViT trên các patch log-mel; SOTA năm 2021. |
| BEATs | Âm thanh tự giám sát | Mô hình của Microsoft, iter3 dẫn đầu AudioSet tính tới 2026. |
| Mixup | Tăng cường theo cặp | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Tăng cường bằng che | Đặt về 0 các dải thời gian và tần số ngẫu nhiên của spectrogram. |
| mAP | Metric đa nhãn chính | Mean average precision trên các lớp và các ngưỡng. |

## Đọc thêm

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — kiến trúc chuẩn mực giai đoạn 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — mặc định từ 2024 trở đi.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — phép tăng cường âm thanh thống trị.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — bộ benchmark 50 lớp vẫn còn sống.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — taxonomy YouTube 632 lớp; vẫn là chuẩn vàng.

---

[← Bài trước: Phổ đồ, thang mel & đặc trưng âm thanh](../../02-spectrograms-mel-features/docs/vi.md) · [Bài sau: Nhận dạng tiếng nói (ASR) →](../../04-speech-recognition-asr/docs/vi.md)
