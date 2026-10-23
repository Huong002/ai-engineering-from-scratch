# LLaVA và tinh chỉnh theo chỉ dẫn thị giác

> LLaVA (tháng 4/2023) là kiến trúc multimodal (đa phương thức) được sao chép nhiều nhất trên hành tinh. Nó thay Q-Former của BLIP-2 bằng một MLP 2 lớp, thay gated cross-attention của Flamingo bằng cách nối token một cách đơn giản, và huấn luyện trên 158k lượt visual instruction tuning (tinh chỉnh theo chỉ dẫn thị giác) do GPT-4 sinh ra từ những chú thích chỉ có văn bản. Bất kỳ kỹ sư nào xây VLM trong giai đoạn 2023–2026 cũng đều xây một biến thể nào đó của LLaVA. LLaVA-1.5 thêm AnyRes. LLaVA-NeXT tăng độ phân giải. LLaVA-OneVision hợp nhất ảnh, nhiều ảnh và video trong một công thức. Bài này đọc công thức đó, cài đặt projector (bộ chiếu), và giải thích vì sao "cách đơn giản đã thắng".

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Giai đoạn 12, Bài 02 (CLIP), Giai đoạn 11 (Kỹ nghệ LLM — tinh chỉnh theo chỉ dẫn)
**Time:** ~180 phút

## Mục tiêu học tập

- Xây một projector MLP 2 lớp ánh xạ embedding patch của ViT (chiều 1024) sang chiều embedding của LLM (chiều 4096).
- Đi qua công thức hai chặng của LLaVA: (1) căn chỉnh projector trên 558k cặp ảnh–chú thích, (2) visual instruction tuning trên 158k lượt do GPT-4 sinh ra.
- Dựng một prompt đúng định dạng LLaVA với placeholder token ảnh, system prompt, và các lượt user/assistant.
- Giải thích vì sao cộng đồng chuyển từ Q-Former sang MLP dù Q-Former thắng về ngân sách token.

## Vấn đề

Q-Former của BLIP-2 (Bài 12.03) nén một ảnh thành 32 token. Gọn, hiệu quả, tốt cho benchmark. Nhưng nó có hai vấn đề.

Thứ nhất, Q-Former huấn luyện được nhưng loss của nó không phải nhiệm vụ cuối. Chặng 1 huấn luyện ITC+ITM+ITG. Chặng 2 huấn luyện LM loss. Các query học một biểu diễn trung gian nào đó mà sau đó LLM phải giải mã. Thông tin bị mất trong nút cổ chai.

Thứ hai, Q-Former tốn 188M tham số, và ở quy mô năm 2023 của LLaVA bạn phải đồng thiết kế nó với LLM đích của mình. Đổi LLM thì huấn luyện lại Q-Former. Đổi vision encoder cũng huấn luyện lại. Mỗi tổ hợp là một dự án R&D riêng.

Câu trả lời của LLaVA đơn giản đến mức ngượng ngùng: lấy 576 token patch của ViT, cho từng token đi qua MLP 2 lớp (`1024 → 4096 → 4096`), rồi đổ cả 576 token vào chuỗi đầu vào của LLM. Không nút cổ chai. Không tiền huấn luyện chặng 1 với những mục tiêu kỳ quặc. Chỉ huấn luyện MLP bằng LM loss trực tiếp.

Dữ liệu từ đâu ra? Hiểu biết thứ hai của LLaVA: dùng GPT-4 (chỉ văn bản) để sinh dữ liệu chỉ dẫn. Đưa cho GPT-4 chú thích COCO và dữ liệu bounding-box của một ảnh, yêu cầu nó tạo hội thoại, mô tả và các câu hỏi suy luận phức tạp. 158k lượt chỉ dẫn–phản hồi miễn phí. Không cần gán nhãn thủ công.

Kết quả: một VLM chạy trên 8 A100 trong một ngày, đánh bại Flamingo trên MMMU, và phát hành một checkpoint mở mà cộng đồng có thể mở rộng. Đến cuối 2023 nó đã sinh ra hơn 50 bản fork.

## Khái niệm

### Kiến trúc

LLaVA-1.5 ở 13B:
- Vision encoder: CLIP ViT-L/14 @ 336 (frozen (đóng băng) ở chặng 1, chặng 2 tuỳ chọn bỏ đóng băng).
- Projector: MLP 2 lớp với kích hoạt GELU, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B (sau này là Llama-3.1-8B).

Lượt lan truyền xuôi trên một ảnh + prompt văn bản:

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

Ảnh chiếm 576 token trong context của LLM. Với context 2048, còn lại 1472 token cho văn bản. Với context 32k, đó chỉ là một sai số làm tròn.

### Chặng 1: alignment stage (chặng căn chỉnh) của projector

Đóng băng ViT. Đóng băng LLM. Chỉ huấn luyện MLP 2 lớp. Dataset: 558k cặp ảnh–chú thích (LAION-CC-SBU). Loss: mô hình hoá ngôn ngữ trên chú thích, với điều kiện là các token ảnh đã được chiếu.

Với một epoch ở batch 128, việc này xong trong vài giờ. Projector học cách ánh xạ không gian ViT sang không gian LLM. Không cần giám sát riêng cho nhiệm vụ.

### Chặng 2: tinh chỉnh theo chỉ dẫn thị giác

Bỏ đóng băng projector (vẫn huấn luyện được). Bỏ đóng băng LLM (thường là toàn bộ, đôi khi dùng LoRA). Huấn luyện trên 158k lượt chỉ dẫn thị giác.

Dữ liệu chỉ dẫn chính là mẹo ở đây. Liu và cộng sự đã tạo nó bằng cách:
1. Lấy một ảnh COCO.
2. Trích mô tả văn bản (5 chú thích của người + danh sách bounding-box).
3. Gửi cho GPT-4 với ba mẫu prompt:
   - Hội thoại: "Tạo một cuộc hội thoại qua lại giữa người dùng và trợ lý về bức ảnh này."
   - Mô tả chi tiết: "Đưa ra một mô tả phong phú, chi tiết về bức ảnh."
   - Suy luận phức tạp: "Hỏi một câu đòi hỏi suy luận về bức ảnh, rồi trả lời câu đó."
4. Phân tích đầu ra của GPT-4 thành các cặp (instruction, response).

Không bước nào chạm trực tiếp vào ảnh — chỉ chạm vào mô tả văn bản. GPT-4 tạo ảo giác ra nội dung ảnh nghe có vẻ hợp lý. Có chút nhiễu, nhưng nó hiệu quả: 158k lượt là đủ để mở khoá hội thoại.

### Vì sao cộng đồng sao chép cách này

- Không có loss riêng cho chặng 1 để phải tinh chỉnh. Dùng LM loss xuyên suốt.
- Projector huấn luyện trong vài giờ, không phải vài ngày.
- Có thể hoán đổi LLM (LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3) bằng cách chỉ huấn luyện lại projector.
- Pipeline dữ liệu chỉ dẫn thị giác dùng GPT-4 và rẻ để tạo lại cho một lĩnh vực mới.

### LLaVA-1.5 và LLaVA-NeXT

LLaVA-1.5 (tháng 10/2023) thêm:
- Dữ liệu nhiệm vụ học thuật (VQA, OKVQA, RefCOCO) trộn vào tinh chỉnh theo chỉ dẫn.
- System prompt tốt hơn.
- 2048 → 32k context.

LLaVA-NeXT (tháng 1/2024) thêm:
- AnyRes: chia ảnh độ phân giải cao thành lưới 2x2 hoặc 1x3 các crop 336x336, cộng thêm một thumbnail toàn cục độ phân giải thấp. Mỗi crop thành 576 token; tổng khoảng 2880 visual token (token thị giác) mỗi ảnh. Các nhiệm vụ OCR và biểu đồ tăng vọt.
- Hỗn hợp dữ liệu chỉ dẫn tốt hơn với ShareGPT4V (chú thích GPT-4V chất lượng cao).
- LLM nền mạnh hơn (Mistral-7B, Yi-34B).

### LLaVA-OneVision

Bài 12.08 đi sâu về OneVision. Bản ngắn: cùng projector, nhưng huấn luyện với một chương trình học bao phủ ảnh đơn, nhiều ảnh và video trong một mô hình với ngân sách token thị giác chia sẻ.

### So sánh với Q-Former

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Số token thị giác mỗi ảnh | 32 | 576 (cơ bản) hoặc 2880 (AnyRes) |
| Tham số huấn luyện được | 188M + LM | 40M + LM |
| Loss chặng 1 | ITC+ITM+ITG | chỉ LM |
| Thay LLM | Cần huấn luyện lại | Hoán đổi với ít huấn luyện lại |
| Nhiều ảnh | Gượng gạo | Tự nhiên (nối) |
| Video | Gượng gạo | Tự nhiên (nối theo từng khung) |
| Ngân sách token | Nhỏ | Lớn |

MLP thắng về sự đơn giản và tính linh hoạt token. Q-Former thắng về ngân sách token. Đến cuối 2023, ngân sách token không còn là ràng buộc cứng (context của LLM đã tăng lên 32k-128k+), và sự đơn giản chiếm ưu thế.

### Định dạng prompt

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` là một token placeholder. Trước khi tokenize, nó được thay bằng 576 token thị giác (hoặc 2880 với AnyRes). Tokenizer thấy một chuỗi dài hơn một chút so với lúc huấn luyện, nhưng LLM vẫn xử lý được đầu vào mới lạ này vì chặng 1 đã dạy nó làm vậy.

### Kinh tế tham số

Phân rã LLaVA-1.5-7B:
- CLIP ViT-L/14 @ 336: 303M (đóng băng ở chặng 1, thường bỏ đóng băng ở chặng 2).
- Projector (2 lớp tuyến tính): ~22M huấn luyện được.
- Llama-7B: 7B.
- Tổng: 7.3B tham số. Huấn luyện được trong chặng 2: toàn bộ 7B + 22M của projector.

Chi phí huấn luyện chặng 2: ~20 giờ trên 8xA100. Đây là con số then chốt — một ngày, một node, tái lập được. Đó là lý do LLaVA lan rộng.

```figure
mm-llava-projector
```

## Áp dụng

`code/main.py` cài đặt:

1. Projector MLP 2 lớp (chiều 16 → 32 → 32 cho quy mô đồ chơi) bằng Python thuần.
2. Pipeline dựng prompt: system prompt + `<image>` được thay bằng N token đã chiếu + lượt user + placeholder cho phần sinh của assistant.
3. Một công cụ trực quan hoá khối thị giác 576 token trông thế nào trong context của LLM (phần trăm context 2k / 32k / 128k bị chiếm).

## Đóng gói

Bài này tạo ra `outputs/skill-llava-vibes-eval.md`. Với một checkpoint thuộc họ LLaVA, nó chạy một bộ vibes-eval 10 prompt (3 mô tả ảnh, 3 VQA, 2 suy luận, 2 từ chối) và báo cáo một bảng điểm dễ đọc cho người. Không phải benchmark; mà là một smoke test để xác nhận projector và LLM đang kết nối tốt với nhau.

## Bài tập

1. Tính số tham số huấn luyện được của projector MLP 2 lớp ở `1024 → 4096 → 4096`. Với GELU và bias, nó chiếm bao nhiêu phần của LLaVA-13B?

2. Dựng một prompt LLaVA cho trường hợp "từ chối" — ảnh chứa một cá nhân riêng tư. Viết phản hồi mong đợi của assistant. Vì sao LLaVA nên từ chối trong tình huống zero-shot này và cần dữ liệu huấn luyện gì để củng cố việc từ chối đó?

3. Đọc phần AnyRes trong blog LLaVA-NeXT. Tính số token thị giác cho một ảnh 1344x672 ở AnyRes. So sánh với 576 token cơ bản ở 336x336.

4. Projector chặng 1 của LLaVA được huấn luyện bằng LM loss trên các chú thích. Điều gì xảy ra nếu bạn bỏ qua chặng 1 và đi thẳng sang chặng 2 (tinh chỉnh theo chỉ dẫn thị giác)? Trích dẫn thí nghiệm loại bỏ của Prismatic VLMs (arXiv:2402.07865) cho câu trả lời.

5. LLaVA-Instruct-150k dùng GPT-4 với chú thích COCO để sinh chỉ dẫn. Với một lĩnh vực mới (X-quang y tế, ảnh vệ tinh), mô tả pipeline dữ liệu bốn bước để sinh chỉ dẫn cho lĩnh vực đó. Điều gì có thể sai ở mỗi bước?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Projector | "cầu MLP" | MLP 2 lớp với GELU ánh xạ chiều ViT sang chiều LLM |
| Image token | "placeholder <image>" | Dấu đánh dấu trong prompt, được thay bằng N token thị giác đã chiếu trước khi suy luận |
| Visual instruction tuning | "chặng 2 của LLaVA" | Huấn luyện trên các bộ ba (ảnh, chỉ dẫn, phản hồi) do GPT-4 sinh ra |
| Stage 1 alignment | "tiền huấn luyện projector" | Đóng băng ViT và LLM, huấn luyện projector bằng LM loss trên chú thích |
| AnyRes | "chia lưới nhiều crop" | Chia ảnh độ phân giải cao thành lưới ô và nối token thị giác của từng ô |
| LLaVA-Instruct | "do GPT-4 sinh ra" | 158k cặp chỉ dẫn–phản hồi được tổng hợp từ chú thích COCO + GPT-4 |
| Vision encoder freeze | "backbone bị khoá" | Trọng số CLIP không cập nhật ở chặng 1, đôi khi cả chặng 2 cũng vậy |
| ShareGPT4V | "chú thích tốt hơn" | 1M chú thích dày đặc do GPT-4V sinh ra, dùng để căn chỉnh chất lượng cao hơn |
| VQA | "hỏi đáp về ảnh" | Nhiệm vụ trả lời một câu hỏi tự do về một bức ảnh |
| Prismatic VLMs | "bài báo về không gian thiết kế" | Thí nghiệm loại bỏ của Karamcheti 2024 kiểm tra có hệ thống các lựa chọn projector và dữ liệu |

## Đọc thêm

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — bài báo LLaVA.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — bộ dữ liệu chú thích dày đặc.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — các thí nghiệm loại bỏ về không gian thiết kế.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — hợp nhất ảnh đơn, nhiều ảnh, video.

---

[← Bài trước: Flamingo và gated cross-attention (cross-attention có cổng) cho VLM few-shot](../../04-flamingo-gated-cross-attention/docs/vi.md) · [Bài sau: Thị giác any-resolution (độ phân giải tuỳ ý): Patch-n'-Pack và NaFlex →](../../06-any-resolution-patch-n-pack/docs/vi.md)
