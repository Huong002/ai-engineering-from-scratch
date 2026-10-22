# Mô hình ngôn ngữ âm thanh — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Năm 2026, các mô hình ngôn ngữ âm thanh suy luận trên cả lời nói + âm thanh môi trường + nhạc. Qwen2.5-Omni-7B ngang GPT-4o Audio trên MMAU-Pro. Audio Flamingo Next vượt Gemini 2.5 Pro trên LongAudioBench. Khoảng cách giữa mã nguồn mở và đóng về cơ bản đã khép lại — trừ các bài multi-audio, nơi mô hình nào cũng gần như đoán ngẫu nhiên.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 04 (ASR), Giai đoạn 12, Bài 03 (Mô hình thị giác - ngôn ngữ), Giai đoạn 7, Bài 10 (Audio Transformer)
**Time:** ~45 phút

## Vấn đề

Bạn có 5 giây âm thanh: chó sủa, ai đó hét "stop!", rồi im lặng. Những câu hỏi hữu ích trải trên nhiều trục:

- **Phiên âm.** "Người ta đã nói gì?" — lãnh địa của ASR.
- **Suy luận ngữ nghĩa.** "Người này có đang gặp nguy hiểm không?" — cần hiểu đồng thời tiếng sủa + tiếng hét + sự im lặng.
- **Suy luận âm nhạc.** "Những nhạc cụ nào đang chơi giai điệu?"
- **Truy xuất âm thanh dài.** "Trong bài giảng 90 phút này, giảng viên giải thích gradient descent ở chỗ nào?"

Một mô hình duy nhất trả lời được tất cả những câu trên chỉ với một prompt chính là **audio language model** (mô hình ngôn ngữ âm thanh) (LALM / ALM). Khác với ASR thuần: LALM sinh câu trả lời ngôn ngữ tự nhiên dạng tự do, không chỉ bản phiên âm.

## Khái niệm

![Mô hình ngôn ngữ âm thanh: encoder âm thanh + projector + decoder LLM](../assets/alm-architecture.svg)

### Khung ba thành phần

Mọi LALM của năm 2026 đều có cùng một bộ xương:

1. **Encoder âm thanh.** Encoder Whisper · BEATs · CLAP · WavLM · hoặc encoder riêng cho từng mô hình.
2. **Projector.** Lớp linear hoặc MLP bắc cầu đặc trưng của encoder âm thanh vào không gian embedding token của LLM.
3. **LLM.** Decoder dựa trên Llama / Qwen / Gemma. Nhận text + audio token (token âm thanh) xen kẽ nhau; sinh ra text.

Huấn luyện:

- **Giai đoạn 1.** Đóng băng encoder + LLM; chỉ huấn luyện projector trên dữ liệu ASR / captioning.
- **Giai đoạn 2.** Fine-tune toàn phần / LoRA trên các bài audio theo chỉ dẫn (QA, suy luận, hiểu nhạc).
- **Giai đoạn 3 (tuỳ chọn).** Voice-in / voice-out thêm một decoder lời nói. Qwen2.5-Omni và AF3-Chat làm theo cách này.

### Bản đồ mô hình 2026

| Mô hình | Backbone | Encoder âm thanh | Phương thức đầu ra | Truy cập |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Kiểm chứng thực tế bằng benchmark (2026)

**MMAU-Pro.** 1800 cặp QA phủ lời nói / âm thanh / nhạc / hỗn hợp. Có cả tập con multi-audio.

| Mô hình | Tổng thể | Lời nói | Âm thanh | Nhạc | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**Cột multi-audio là lời cảnh tỉnh cho tất cả mọi người.** Đoán ngẫu nhiên với câu hỏi trắc nghiệm 4 lựa chọn = 25%; hầu hết mô hình chỉ quanh quẩn mức đó. LALM vẫn còn chật vật khi phải so sánh hai đoạn clip.

### LALM hữu ích ở đâu trong năm 2026

- **Kiểm toán tuân thủ bản ghi tổng đài.** "Điện thoại viên có đọc đúng phần công bố bắt buộc không?"
- **Khả năng tiếp cận.** Mô tả sự kiện âm thanh cho người khiếm thính (không chỉ phiên âm).
- **Kiểm duyệt nội dung.** Phát hiện ngôn từ bạo lực + giọng điệu đe doạ + ngữ cảnh nền.
- **Chia chương podcast / cuộc họp.** Tóm tắt theo ngữ nghĩa, không chỉ theo lượt nói.
- **Phân tích danh mục nhạc.** "Tìm mọi bản nhạc có chuyển giọng ở đoạn B."

### Nơi chúng CHƯA hữu ích

- Lý thuyết âm nhạc chi tiết (dưới mức hợp âm).
- Suy luận quy công người nói trên các cuộc hội thoại dài (suy giảm sau 10 phút).
- So sánh multi-audio (22-26% chỉ nhỉnh hơn mức ngẫu nhiên).
- Suy luận truyền trực tiếp (streaming) theo thời gian thực (hầu hết vẫn là suy luận batch ngoại tuyến).

```figure
v4-alm-tokens
```

## Tự tay làm

### Bước 1: truy vấn Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Bước 2: mẫu projector

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Chỉ có thế. Projector (bộ nối) thường là 1-3 lớp linear. Huấn luyện nó trên các cặp ASR (audio → bản phiên âm) chính là bài tiền huấn luyện của Giai đoạn 1.

### Bước 3: chấm điểm trên MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("gamma-lab-umd/MMAU-Pro", split="test")
mcq = mmau.filter(lambda item: len(item["choices"] or []) > 1)

correct = 0
for item in mcq:
    answer = call_model(item["audio_path"], item["question"], item["choices"])
    if answer == item["answer"]:
        correct += 1
print(f"Accuracy: {correct / len(mcq):.3f}")
```

`audio_path` trỏ vào `data.zip` của repo dataset (khoảng 47 GB), nên hãy tải và giải nén trước khi chấm. Vòng so khớp chính xác này chỉ để kiểm tra nhanh, không phải scorer của benchmark, nên con số nó cho không so sánh được với kết quả MMAU-Pro đã công bố. Bộ đánh giá chính thức khớp đáp án trắc nghiệm theo độ tương đồng embedding (NV-Embed-v2), chấm câu trả lời mở bằng một LLM judge, và kiểm tra câu trả lời theo chỉ dẫn bằng luật regex: ghi dự đoán vào cột `model_output` rồi chạy `evaluate_mmau_pro_comprehensive.py` từ [repo MMAU-Pro](https://github.com/sonalkum/MMAUPro). Hãy báo cáo riêng từng `category` (lời nói, âm thanh, nhạc, multi, và phần còn lại). Con số tổng hợp sẽ che mất chỗ mô hình thất bại.

## Áp dụng

| Nhiệm vụ | Lựa chọn năm 2026 |
|------|-----------|
| Hỏi đáp audio tự do (mở) | Qwen2.5-Omni-7B |
| Mở, tốt nhất trên audio dài | Audio Flamingo Next |
| Đóng, tốt nhất | Gemini 2.5 Pro |
| Tác tử voice-in / voice-out | Qwen2.5-Omni hoặc GPT-4o Audio |
| Suy luận âm nhạc | Audio Flamingo 3 hoặc 2 (AF-CLAP chuyên nhạc) |
| Kiểm toán tổng đài | Gemini 2.5 Pro qua API, kèm RAG trên tài liệu chính sách của bạn |

## Cạm bẫy

- **Tin quá mức vào multi-audio.** Nếu bài toán của bạn là "clip nào có X", thì mức hiệu năng ngang đoán ngẫu nhiên là có thật.
- **Suy giảm trên audio dài.** Quá 10 phút, việc quy công người nói của hầu hết mô hình bị vỡ. Hãy tách người nói trước (Bài 6), rồi mới tóm tắt.
- **Ảo giác trên khoảng lặng.** Đúng vấn đề kiểu Whisper mà các LALM dùng encoder Whisper thừa hưởng. Hãy chặn bằng VAD.
- **Chọn benchmark có lợi.** Blog của nhà cung cấp thường chỉ nêu hạng mục đẹp nhất. Hãy tự chạy tập con multi-audio của MMAU-Pro.

## Đóng gói

Lưu thành `outputs/skill-alm-picker.md`. Chọn LALM + tập con benchmark + phương thức đầu ra (text hay speech) cho một bài toán hiểu âm thanh cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py` để xem một mẫu projector đồ chơi + cách LALM giả lập định tuyến (audio-embedding, text-tokens) → token đầu ra.
2. **Trung bình.** Chấm Qwen2.5-Omni-7B trên 100 mục lời nói của MMAU-Pro. So với con số bài báo công bố.
3. **Khó.** Dựng một baseline mô tả âm thanh tối giản: encoder BEATs + projector 2 lớp + Llama-3.2-1B đóng băng. Chỉ fine-tune projector trên AudioCaps. So với SALMONN trên Clotho-AQA.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Encoder âm thanh + projector + decoder LLM. |
| Projector | Bộ nối (adapter) | MLP nhỏ ánh xạ đặc trưng âm thanh vào không gian embedding của LLM. |
| MMAU | Cái benchmark | 10k cặp hỏi đáp audio phủ lời nói, âm thanh, nhạc. |
| MMAU-Pro | MMAU khó hơn | 1800 câu hỏi multi-audio / nặng suy luận. |
| LongAudioBench | Đánh giá dạng dài | Clip dài nhiều phút kèm truy vấn ngữ nghĩa. |
| Voice-in / voice-out | Speech-native | Mô hình nhận lời nói và phát ra lời nói, không vòng qua text. |

## Đọc thêm

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — kiến trúc tham chiếu.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — đầu bảng audio dài trong nhóm mở.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — người tiên phong kiến trúc hai encoder.
- [MMAU-Pro leaderboard](https://sonalkum.github.io/mmau-pro/) — bảng xếp hạng trực tiếp năm 2026.

---

[← Bài trước: Sinh nhạc (music generation)](../../09-music-generation/docs/vi.md) · [Bài sau: Xử lý âm thanh trong thời gian thực →](../../11-real-time-audio-processing/docs/vi.md)
