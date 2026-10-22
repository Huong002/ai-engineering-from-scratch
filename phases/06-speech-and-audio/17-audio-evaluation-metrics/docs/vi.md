# Đánh giá âm thanh — WER, MOS, UTMOS, MMAU, FAD và các bảng xếp hạng mở

> Bạn không thể ship thứ mình không đo được. Bài này gọi tên các chỉ số của năm 2026 cho từng tác vụ âm thanh: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER vòng qua ASR), âm thanh-ngôn ngữ (MMAU, LongAudioBench), nhạc (FAD, CLAP) và người nói (EER). Cùng với những bảng xếp hạng để bạn so sánh.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 04, 06, 07, 09, 10; Giai đoạn 2, Bài 09 (Đánh giá mô hình)
**Time:** ~60 phút

## Vấn đề

Mỗi tác vụ âm thanh có nhiều chỉ số, mỗi chỉ số đo một trục khác nhau. Dùng sai chỉ số là cách bạn ship một mô hình trông tuyệt vời trên dashboard và tệ hại trong production. Danh sách chuẩn của 2026:

| Tác vụ | Chính | Phụ |
|------|---------|-----------|
| ASR | WER | CER · RTFx · độ trễ token đầu |
| TTS | MOS / UTMOS | SECS · WER vòng qua ASR · CER · TTFA |
| Cloning giọng nói | SECS (cosine ECAPA) | MOS · CER |
| Xác minh người nói | EER | minDCF · FAR / FRR tại điểm vận hành |
| Phân tách người nói | DER | JER · nhầm lẫn người nói |
| Phân loại âm thanh | top-1 · mAP | macro F1 · recall theo lớp |
| Sinh nhạc | FAD | CLAP · MOS từ hội đồng nghe |
| Mô hình ngôn ngữ âm thanh | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| S2S streaming | độ trễ P50/P95 | WER · MOS |

## Khái niệm

![Ma trận đánh giá âm thanh — chỉ số vs tác vụ vs bảng xếp hạng 2026](../assets/eval-landscape.svg)

### Chỉ số ASR

**WER (word error rate, tỉ lệ lỗi từ).** `(S + D + I) / N`. Chuyển về chữ thường, bỏ dấu câu, chuẩn hoá số trước khi chấm điểm. Dùng `jiwer` hoặc `whisper_normalizer` của OpenAI. &lt; 5% = ngang trình người với giọng đọc.

**CER (character error rate, tỉ lệ lỗi ký tự).** Cùng công thức nhưng ở mức ký tự. Dùng cho ngôn ngữ có thanh điệu (Quan thoại, Quảng Đông) nơi việc tách từ là mơ hồ.

**RTFx (inverse real-time factor).** Số giây âm thanh xử lý được trên mỗi giây đồng hồ thực. Càng cao càng tốt. Parakeet-TDT đạt 3380×. Whisper-large-v3 khoảng ~30×.

**Độ trễ token đầu.** Thời gian đồng hồ thực từ lúc có âm thanh đầu vào tới token bản ghi đầu tiên. Quan trọng với streaming. Deepgram Nova-3: ~150 ms.

### Chỉ số TTS

**MOS (mean opinion score, điểm chất lượng cảm nhận).** Thang điểm 1-5 do con người chấm. Chuẩn vàng nhưng chậm. Thu 20+ người nghe cho mỗi mẫu, 100+ mẫu cho mỗi mô hình.

**UTMOS (2022-2026).** Mô hình dự đoán MOS đã học. Tương quan ~0.9 với MOS của người trên các benchmark chuẩn. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** Dùng cho cloning giọng, đo độ giống giọng. Cosine embedding ECAPA giữa bản tham chiếu và bản đầu ra clone. &gt; 0.75 = bản clone còn nhận ra được.

**WER vòng qua ASR.** Chạy Whisper trên đầu ra TTS, tính WER so với văn bản đầu vào. Bắt được các thoái hoá về độ dễ hiểu. SOTA 2026: &lt; 2% CER.

**TTFA (time-to-first-audio).** Độ trễ đồng hồ thực. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### Riêng cho cloning giọng nói

**SECS + MOS + CER** như một bộ ba. Bản clone có SECS cao nhưng MOS thấp nghĩa là đúng âm sắc nhưng thiếu tự nhiên; ngược lại là giọng tự nhiên nhưng sai người nói.

### Xác minh người nói

**EER (Equal Error Rate, tỉ lệ lỗi cân bằng).** Ngưỡng mà tại đó tỉ lệ chấp nhận sai bằng tỉ lệ từ chối sai. ECAPA trên VoxCeleb1-O: 0.87%.

**minDCF (min Detection Cost).** Chi phí có trọng số tại một điểm vận hành đã chọn (thường FAR=0.01). Sát với production hơn EER.

### Phân tách người nói

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Giọng bị bỏ sót + giọng báo nhầm + nhầm lẫn người nói, mỗi phần như một tỉ lệ. Với các cuộc họp AMI: DER ~10-20% là thực tế. pyannote 3.1 + Precision-2 bản thương mại: &lt;10% DER trên âm thanh ghi tốt.

**JER (Jaccard Error Rate).** Phương án thay thế DER, bền hơn với thiên lệch do đoạn ngắn.

### Phân loại âm thanh

Đa nhãn: **mAP (mean Average Precision)** trên tất cả các lớp. AudioSet: 0.548 mAP cho BEATs-iter3.

Đa lớp loại trừ nhau: **độ chính xác top-1, top-5**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

Mất cân bằng: **macro F1** + **recall theo lớp**. Báo cáo từng lớp — độ chính xác gộp che mất lớp nào đang hỏng.

### Sinh nhạc

**FAD (Fréchet Audio Distance).** Khoảng cách giữa phân phối embedding VGGish của âm thanh thật và âm thanh sinh ra. MusicGen-small trên MusicCaps: 4.5. MusicLM: 4.0. Càng thấp càng tốt.

**CLAP Score.** Điểm khớp văn bản-âm thanh dùng embedding CLAP. &gt; 0.3 = khớp ở mức chấp nhận được.

**MOS từ hội đồng nghe.** Vẫn là tiếng nói cuối cùng với nhạc cho người dùng phổ thông. Suno v5 đạt ELO 1293 trên TTS Arena (từ so sánh cặp do con người bình chọn).

### Benchmark âm thanh-ngôn ngữ

**MMAU (Massive Multi-Audio Understanding).** 10 nghìn cặp hỏi-đáp về âm thanh.

**MMAU-Pro.** 1800 câu khó, bốn nhóm: giọng nói / âm thanh / nhạc / đa âm thanh. Đoán ngẫu nhiên được 25% với câu 4 lựa chọn. Gemini 2.5 Pro tổng thể ~60%; riêng đa âm thanh ~22% ở mọi mô hình.

**LongAudioBench.** Clip dài nhiều phút kèm truy vấn ngữ nghĩa. Audio Flamingo Next vượt Gemini 2.5 Pro.

**AudioCaps / Clotho.** Benchmark tạo mô tả. Các chỉ số SPICE, CIDEr, FENSE.

### Giọng nói sang giọng nói streaming

**Độ trễ P50 / P95 / P99.** Thời gian đồng hồ thực từ lúc người dùng nói xong tới khi có phản hồi nghe được đầu tiên. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**WER / MOS** trên đầu ra.

**Khả năng chen lời.** Thời gian từ lúc người dùng ngắt lời tới khi trợ lý im. Mục tiêu &lt; 150 ms.

### Bảng xếp hạng 2026

| Bảng xếp hạng | Hạng mục | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | Tiếng Anh + đa ngữ + dạng dài | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | TTS tiếng Anh | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO từ bình chọn cặp | `artificialanalysis.ai/speech` |
| MMAU-Pro | Suy luận LALM | `sonalkum.github.io/mmau-pro` |
| SpeakerBench / VoxSRC | Nhận dạng người nói | `voxsrc.github.io` |
| MMAU music subset | LALM nhạc | (nằm trong MMAU) |
| HEAR benchmark | Âm thanh tự giám sát | `hearbenchmark.com` |

```figure
sp-wer-align
```

## Tự tay làm

### Bước 1: WER với chuẩn hoá

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Bước 2: WER vòng TTS

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Bước 3: SECS cho cloning giọng

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Bước 4: FAD cho sinh nhạc

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Bước 5: EER cho xác minh người nói (cùng code như Bài 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Áp dụng

Ghép mỗi lần deploy với một harness đánh giá cố định, chạy mỗi khi mô hình cập nhật. Ba nguyên tắc cốt lõi:

1. **Chuẩn hoá trước khi chấm điểm.** Chữ thường, bỏ dấu câu, mở số. Báo cáo cả quy tắc chuẩn hoá.
2. **Báo cáo phân phối, không phải trung bình.** P50/P95/P99 cho độ trễ. Recall theo lớp cho phân loại. Theo nhóm cho MMAU.
3. **Chạy một benchmark công khai chuẩn.** Dù dữ liệu production của bạn khác, báo cáo trên Open ASR / TTS Arena / MMAU giúp người đánh giá so sánh cùng hệ quy chiếu.

## Cạm bẫy

- **Ngoại suy UTMOS.** Huấn luyện trên giọng sạch kiểu VCTK; chấm kém với âm thanh nhiễu / clone / nhiều cảm xúc.
- **Thiên lệch hội đồng MOS.** 20 công nhân Amazon Mechanical Turk ≠ 20 người dùng mục tiêu. Nếu rủi ro cao, hãy trả tiền cho hội đồng đúng lĩnh vực.
- **FAD phụ thuộc tập tham chiếu.** Khi so sánh giữa các mô hình, phải dùng cùng một phân phối tham chiếu.
- **WER gộp.** WER 5% tổng thể có thể che đi 30% WER trên giọng có trọng âm. Báo cáo theo từng nhóm nhân khẩu.
- **Bão hoà benchmark công khai.** Hầu hết mô hình biên đều gần trần trên các benchmark chuẩn. Hãy xây một tập giữ riêng nội bộ phản ánh lưu lượng thật của bạn.

## Đóng gói

Lưu thành `outputs/skill-audio-evaluator.md`. Chọn chỉ số, benchmark và định dạng báo cáo cho mọi lần phát hành mô hình âm thanh.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Tính WER / CER / EER / SECS / kiểu FAD / kiểu MMAU trên dữ liệu đồ chơi.
2. **Trung bình.** Xây harness WER vòng TTS. Cho đầu ra Kokoro hoặc F5-TTS của bạn đi qua Whisper. Tính WER trên 50 prompt. Đánh dấu những prompt có WER &gt; 10%.
3. **Khó.** Chấm lựa chọn LALM của Bài 10 trên các nhóm giọng nói + đa âm thanh của MMAU-Pro (mỗi nhóm 50 mục). Báo cáo độ chính xác theo nhóm và so với con số đã công bố.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| WER | Điểm ASR | `(S+D+I)/N` ở mức từ sau khi chuẩn hoá. |
| CER | WER ở mức ký tự | Dành cho ngôn ngữ có thanh điệu hoặc hệ thống mức ký tự. |
| MOS | Ý kiến con người | Thang 1-5; 20+ người nghe × 100 mẫu. |
| UTMOS | Bộ dự đoán MOS bằng ML | Mô hình đã học; tương quan ~0.9 với MOS của người. |
| SECS | Độ giống bản clone | Cosine ECAPA giữa bản tham chiếu và bản clone. |
| EER | Điểm xác minh người nói | Ngưỡng tại đó FAR = FRR. |
| DER | Điểm phân tách người nói | (FA + Miss + Confusion) / tổng. |
| FAD | Chất lượng sinh nhạc | Khoảng cách Fréchet trên embedding VGGish. |
| RTFx | Thông lượng | Số giây âm thanh trên mỗi giây đồng hồ thực. |

## Đọc thêm

- [jiwer](https://github.com/jitsi/jiwer) — thư viện WER/CER kèm tiện ích chuẩn hoá.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — bộ dự đoán MOS đã học.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — chuẩn cho sinh nhạc.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — bảng xếp hạng trực tiếp 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — bảng xếp hạng TTS theo bình chọn của người.
- [MMAU-Pro benchmark](https://sonalkum.github.io/mmau-pro/) — bảng xếp hạng suy luận LALM.
- [HEAR benchmark](https://hearbenchmark.com/) — benchmark âm thanh tự giám sát.

---

[← Bài trước: Chống giả mạo giọng nói & dấu thuỷ văn âm thanh](../../16-anti-spoofing-audio-watermarking/docs/vi.md)
