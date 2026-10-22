# Nhận dạng tiếng nói (ASR) — CTC, RNN-T, Attention

> Nhận dạng tiếng nói (speech recognition) là phân loại âm thanh ở mọi bước thời gian, được nối lại bằng một mô hình chuỗi biết tiếng Anh và biết khoảng lặng. CTC, RNN-T và attention là ba cách để làm việc đó. Hãy chọn một cách và hiểu vì sao.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Spectrogram & Mel), Giai đoạn 5, Bài 08 (CNN & RNN cho văn bản), Giai đoạn 5, Bài 10 (Cơ chế attention)
**Time:** ~45 phút

## Vấn đề

Bạn có một clip 10 giây ở 16 kHz. Bạn muốn một chuỗi: "turn on the kitchen lights". Thử thách nằm ở cấu trúc: các frame âm thanh không căn một-một với các ký tự. Từ "okay" có thể kéo dài 200 ms hoặc 1200 ms. Khoảng lặng chen vào câu nói. Có những âm vị dài hơn âm vị khác. Số token đầu ra không biết trước.

Ba cách tiếp cận giải quyết việc này:

1. **CTC (Connectionist Temporal Classification — phân loại thời gian kết nối).** Phát xác suất token cho từng frame, gồm một *blank* đặc biệt. Khi giải mã thì gộp các lần lặp và bỏ blank. Không tự hồi quy (non-autoregressive), nhanh. Được dùng bởi wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Mạng joint dự đoán token tiếp theo từ frame của encoder và các token trước. Có thể stream. Được dùng bởi ASR on-device của Google, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Encoder nén âm thanh thành hidden state, decoder cross-attend để sinh token tự hồi quy. Được dùng bởi Whisper, SeamlessM4T.

Năm 2026, WER SOTA trên LibriSpeech test-clean là 1.4% (Parakeet-TDT-1.1B, NVIDIA) và 1.58% (Whisper-Large-v3-turbo). Khác biệt rất nhỏ; khác biệt khi triển khai thì rất lớn.

## Khái niệm

![Ba cách tiếp cận ASR: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Trực giác CTC.** Cho encoder xuất ra `T` phân phối theo từng frame trên `V+1` token (V ký tự + blank). Với chuỗi đích `y` có độ dài `U < T`, mọi cách căn frame mà khi gộp lại thành `y` đều được tính. Loss CTC lấy tổng trên tất cả các cách căn đó. Suy luận: argmax từng frame, gộp các lần lặp, bỏ blank.

Ưu điểm: không tự hồi quy, có thể stream, không nhìn trước. Nhược điểm: *giả định độc lập có điều kiện* — mỗi dự đoán frame độc lập với các frame khác, nên không có language model (mô hình ngôn ngữ) bên trong. Khắc phục bằng LM ngoài qua beam search (tìm kiếm chùm) hoặc shallow fusion.

**Trực giác RNN-T.** Thêm một mạng *predictor* nhúng lịch sử token và một *joiner* kết hợp trạng thái predictor với frame của encoder thành phân phối joint trên `V+1` (`+1` là null / không phát gì). Nó mô hình hoá rõ ràng sự phụ thuộc có điều kiện mà CTC đã bỏ qua. Có thể stream vì mỗi bước chỉ phụ thuộc vào các frame trước và các token trước.

Ưu điểm: stream được + có LM bên trong. Nhược điểm: huấn luyện phức tạp hơn và tốn bộ nhớ hơn (lưới loss 3D); các kernel loss RNN-T tự nó đã là cả một hạng mục thư viện riêng.

**Attention encoder-decoder.** Encoder (6-32 lớp transformer) trên các frame log-mel. Decoder (6-32 lớp transformer) cross-attend vào đầu ra encoder để sinh token tự hồi quy. Không có ràng buộc căn chỉnh — attention có thể nhìn bất cứ đâu trong âm thanh. Không stream được trừ khi bạn giới hạn attention (chunked Whisper-Streaming, 2024).

Ưu điểm: chất lượng cao nhất cho ASR offline, dễ huấn luyện bằng bộ công cụ seq2seq chuẩn. Nhược điểm: độ trễ tự hồi quy tỉ lệ với độ dài đầu ra; không stream được nếu không làm thêm kỹ thuật.

### WER: một con số duy nhất

**Word Error Rate (tỉ lệ lỗi từ)** = `(S + D + I) / N`, trong đó S=thay thế, D=xoá, I=thêm, N=số từ tham chiếu. Tương đương khoảng cách sửa Levenshtein ở mức từ. Càng thấp càng tốt. WER trên 20% nói chung là không dùng được; dưới 5% là ngang người cho lời nói đọc. Số liệu 2026 trên các benchmark chuẩn:

| Mô hình | LibriSpeech test-clean | LibriSpeech test-other | Kích thước |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Tất cả các mô hình này đều dựa trên encoder-decoder hoặc RNN-T. Các hệ CTC thuần (wav2vec 2.0) nằm quanh 1.8–2.1% trên test-clean.

```figure
ctc-collapse
```

## Tự tay làm

### Bước 1: giải mã CTC greedy

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Hai quy tắc: gộp các lần lặp liên tiếp, bỏ blank. Ví dụ: `a a _ _ a b b _ c` → `a a b c`.

### Bước 2: beam search cho CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Hệ production dùng prefix tree beam search có fusion LM; đây chỉ là bộ khung khái niệm.

### Bước 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Bước 4: suy luận với Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Một dòng lệnh cho hệ ASR tổng quát mạnh nhất năm 2026. Chạy trên GPU 24 GB với tốc độ ~20× thời gian thực.

### Bước 5: streaming với Parakeet hoặc wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR cần attention theo chunk ở encoder và trạng thái mang sang; hãy dùng thư viện hỗ trợ việc này (NeMo cho Parakeet, pipeline `transformers` với `chunk_length_s`).

## Áp dụng

Bộ stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Tiếng Anh, offline, chất lượng tối đa | Whisper-large-v3-turbo |
| Đa ngôn ngữ, bền | SeamlessM4T v2 |
| Streaming, độ trễ thấp | Parakeet-TDT-1.1B hoặc Riva |
| Edge, di động, độ trễ <500 ms | Whisper-Tiny lượng tử hoá hoặc Moonshine (2024) |
| Dạng dài | Whisper với chunking dựa trên VAD (WhisperX) |
| Chuyên ngành (y tế, pháp lý) | Fine-tune wav2vec 2.0 + fusion LM chuyên ngành |

## Những cạm bẫy vẫn còn trong năm 2026

- **Không có VAD.** Chạy Whisper trên khoảng lặng sinh ra ảo giác ("Thanks for watching!"). Luôn chặn bằng VAD.
- **WER theo ký tự vs theo từ vs theo subword.** Hãy báo WER mức từ *sau* khi chuẩn hoá (viết thường, bỏ dấu câu).
- **Lệch nhận dạng ngôn ngữ.** LID tự động của Whisper định tuyến sai các clip nhiễu sang tiếng Nhật hoặc tiếng Wales; hãy ép `language="en"` khi bạn đã biết.
- **Clip dài không chia chunk.** Whisper có cửa sổ 30 giây. Dùng `chunk_length_s=30, stride=5` cho bất cứ thứ gì dài hơn.

## Đóng gói

Lưu thành `outputs/skill-asr-picker.md`. Chọn model, chiến lược giải mã, chunking và fusion LM cho một mục tiêu triển khai cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó giải mã greedy một đầu ra CTC thủ công và tính WER so với bản tham chiếu.
2. **Trung bình.** Cài đặt tử tế prefix-tree beam search ở Bước 2 (xử lý đúng quy tắc gộp blank). So sánh với greedy trên một bộ dữ liệu tổng hợp 10 ví dụ.
3. **Khó.** Dùng `whisper-large-v3-turbo` trên [LibriSpeech test-clean](https://www.openslr.org/12). Tính WER trên 100 câu đầu. So sánh với các con số đã công bố.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| CTC | Loss của token blank | Lấy biên trên tất cả các cách căn frame-với-token; không tự hồi quy. |
| RNN-T | Loss để stream | CTC + predictor token kế tiếp; xử lý được thứ tự từ. |
| Attention enc-dec | Kiểu Whisper | Encoder + decoder cross-attend; chất lượng offline tốt nhất. |
| WER | Con số bạn báo cáo | `(S+D+I)/N` ở mức từ. |
| Blank | Sự trống rỗng | Token đặc biệt trong CTC báo hiệu "frame này không phát gì". |
| LM fusion | Mô hình ngôn ngữ ngoài | Cộng log-prob của LM có trọng số khi beam search. |
| VAD | Cổng chặn khoảng lặng | Voice activity detector; cắt phần không phải lời nói. |

## Đọc thêm

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — bài báo CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — bài báo RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — bài báo chuẩn năm 2022; phần mở rộng v3-turbo năm 2024.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — dẫn đầu Open ASR Leaderboard 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — benchmark trực tiếp trên 25+ mô hình.

---

[← Bài trước: Phân loại âm thanh](../../03-audio-classification/docs/vi.md) · [Bài sau: Whisper →](../../05-whisper-architecture-finetuning/docs/vi.md)
