# MIO và các mô hình đa phương thức any-to-any (bất kỳ sang bất kỳ) truyền trực tiếp

> GPT-4o tung ra một sản phẩm mà hầu hết mô hình mở không sao chép nổi: một agent nghe được giọng nói, nhìn được video và đáp lại bằng giọng nói trong thời gian thực. Câu trả lời của hệ sinh thái mở vào cuối 2024 là MIO (Wang và cộng sự, tháng 9 năm 2024). MIO tokenize văn bản, hình ảnh, giọng nói và âm nhạc, huấn luyện một transformer nhân quả trên các chuỗi interleaved (xen kẽ), rồi sinh ra bất kỳ phương thức nào từ bất kỳ phương thức nào. AnyGPT (Zhan và cộng sự, tháng 2 năm 2024) là bản chứng minh khái niệm; MIO là bản mở rộng quy mô; Unified-IO 2 (Allen AI, tháng 12 năm 2023) là người anh em họ có thêm nền tảng thị giác + hành động. Bài này đọc mẫu any-to-any — bốn tokenizer, một transformer, giải mã thân thiện với streaming (truyền trực tiếp).

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Giai đoạn 12, Bài 11 (Chameleon), Giai đoạn 6 (Giọng nói và âm thanh)
**Time:** ~120 phút

## Mục tiêu học tập

- Thiết kế một từ vựng dùng chung chứa được token văn bản, hình ảnh, giọng nói và âm nhạc mà không xung đột.
- So sánh SEED-Tokenizer (hình ảnh) với residual-VQ của SpeechTokenizer (giọng nói) về đánh đổi nén và tái tạo.
- Giải thích chương trình bốn giai đoạn xây dựng năng lực sinh any-to-any.
- Kể tên ba công thức any-to-any mở và đánh đổi chính của chúng: MIO, AnyGPT, Unified-IO 2.

## Vấn đề

Một mô hình đa phương thức thống nhất thì dễ tuyên bố và khó xây ở quy mô lớn. Hầu hết hệ "any-to-any" cho tới 2024 đều chạy theo kiểu nối chuỗi: mô hình thị giác → biểu diễn văn bản → mô hình giọng nói → âm thanh. Mỗi chặng đều mất thông tin, thêm latency (độ trễ) và làm việc huấn luyện phức tạp hơn. Video demo của GPT-4o cho thấy một phương án một-mô-hình với phản hồi dưới một giây; các hệ mở chạy sau hàng tháng.

Các thách thức kỹ thuật:

- Tokenizer phải có cho mọi phương thức, nén đủ ít mất mát để tái tạo, và sinh token ở tốc độ mà transformer tiêu thụ được.
- Một từ vựng duy nhất phải phân bổ chỗ cho văn bản (32k+), hình ảnh (16k+), giọng nói (4k+), âm nhạc (8k+). Tối thiểu hơn bốn mươi nghìn mục.
- Dữ liệu huấn luyện phải phủ mọi cặp đầu vào - đầu ra (văn bản→hình ảnh, hình ảnh→giọng nói, giọng nói→hình ảnh, v.v.) hoặc mô hình phải biết tổ hợp.
- Suy luận phải truyền token đầu ra đủ nhanh cho độ trễ hội thoại (<500ms thời gian tới byte âm thanh đầu tiên).

## Khái niệm

### Bốn tokenizer cho bốn phương thức

Chồng tokenizer của MIO:

- Văn bản: BPE chuẩn, từ vựng ~32000.
- Hình ảnh: SEED-Tokenizer (2023) — VAE lượng tử hoá với codebook rời rạc, 4096 mục, 32x32 token mỗi ảnh.
- Giọng nói: residual-VQ của SpeechTokenizer (2023) — mã hoá dạng sóng 16kHz thành 8 codebook phân cấp; tầng đầu là nội dung thô, các tầng sau thêm ngữ điệu và danh tính người nói.
- Âm nhạc: residual-VQ tương tự (họ MusicGen / Encodec của Meta), 4-8 codebook.

Mỗi phương thức sinh ra token số nguyên. Các token này nhận những dải ID rời nhau trong từ vựng dùng chung:

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

Tổng cộng: từ vựng ~48k. Embedding đầu vào và phép chiếu đầu ra phủ toàn bộ từ vựng đó.

### Giải mã truyền trực tiếp

Sinh giọng nói dùng residual-VQ. Transformer dự đoán các speech token (token giọng nói) ở tầng cơ sở (layer 0); một bộ lượng tử hoá residual giải mã song song sẽ dự đoán các tầng tiếp theo. Mỗi token layer 0 tương ứng khoảng 50ms âm thanh ở 16kHz.

Mẫu truyền trực tiếp:

1. Người dùng nói vào mic; tokenizer âm thanh thời gian thực phát token giọng nói mỗi 50ms.
2. MIO tiêu thụ token ngay khi chúng tới (prefill prompt + lan truyền tăng dần).
3. Token đầu ra được truyền ra ngay khi sinh; một bộ giải mã giọng nói song song chuyển chúng thành mẫu âm thanh với độ trễ ~50-150ms.
4. Thời gian tới byte âm thanh đầu tiên: ~300-500ms trong bài báo MIO, tiến gần mức ~250ms của GPT-4o.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612) và Moshi (arXiv:2410.00037) là những thiết kế speech-LLM truyền trực tiếp bổ trợ nhau. Moshi đặc biệt đạt vòng phản hồi 160ms trên một GPU.

### Chương trình bốn giai đoạn

Chương trình huấn luyện của MIO:

1. Giai đoạn 1 — căn chỉnh. Kho ngữ liệu cặp phương thức quy mô lớn: văn bản-hình ảnh, văn bản-giọng nói, văn bản-âm nhạc. Mỗi cặp dùng đoạn từ vựng token riêng. Huấn luyện từ vựng dùng chung.
2. Giai đoạn 2 — interleaved. Tài liệu đa phương thức xen kẽ (blog có ảnh + video, podcast kèm bản ghi, v.v.). Huấn luyện ngữ cảnh liên phương thức.
3. Giai đoạn 3 — tăng cường giọng nói. Thêm dữ liệu âm thanh để nâng chất lượng giọng nói mà không mất năng lực văn bản.
4. Giai đoạn 4 — SFT. Tinh chỉnh theo chỉ dẫn trên nhiều phương thức: VQA, tạo chú thích, tường thuật, hội thoại giọng nói sang giọng nói.

Thiếu một giai đoạn sẽ làm suy giảm những năng lực cụ thể: bỏ giai đoạn 2 thì mô hình mất ngữ cảnh liên phương thức; bỏ giai đoạn 3 thì giọng nói kém.

### Chuỗi tư duy thị giác (chain-of-visual-thought)

MIO giới thiệu chain-of-visual-thought: mô hình phát ra các token hình ảnh trung gian như một bước suy luận. Với câu hỏi "con mèo có đang trèo cây không?", mô hình:

1. Phát ra token `<image>` dựng lại khung cảnh (từ ảnh đầu vào hoặc từ một bản phác).
2. Phát ra văn bản phân tích bản phác đó.
3. Phát ra câu trả lời cuối cùng.

Hình ảnh trung gian được dựng ra đóng vai trò bảng nháp. Điểm chuẩn cải thiện trên các bài toán suy luận không gian. Ý tưởng này phản chiếu chain-of-thought cho suy luận văn bản.

### Các đối thủ trong any-to-any

- AnyGPT (arXiv:2402.12226): 4 phương thức (văn bản, hình ảnh, giọng nói, âm nhạc), thiết kế tương tự.
- Unified-IO 2 (arXiv:2312.17172): thêm đầu ra hành động thị giác, độ sâu, pháp tuyến. Đa dạng tác vụ hơn, quy mô nhỏ hơn.
- NExT-GPT (arXiv:2309.05519): LLM + các bộ giải mã diffusion riêng cho từng phương thức. Không phải cách tiếp cận một-mô-hình.
- CoDi (arXiv:2305.11846): diffusion tổ hợp được; any-to-any qua latent dùng chung.

MIO gần nhất với any-to-any thuần token. AnyGPT là tổ tiên về ý tưởng của nó.

### Ngân sách độ trễ

Với một sản phẩm hội thoại, độ trễ của từng thành phần đều quan trọng:

- Mic tới token âm thanh: ~50ms.
- Prefill (token âm thanh + lịch sử): ~100ms trên mô hình 8B.
- Token đầu ra đầu tiên: ~50ms.
- Residual-VQ song song + bộ giải mã giọng nói: ~100-150ms.

Tổng thời gian tới byte âm thanh đầu tiên: tối thiểu ~300ms. GPT-4o tuyên bố ~250ms. Moshi tuyên bố 160ms. MIO/AnyGPT nằm trong khoảng 400-600ms theo các điểm chuẩn công khai.

### Vì sao any-to-any vẫn khó

Dù đã sang 2026, các mô hình any-to-any mở vẫn kém mô hình đóng ở hai trục:

- Chất lượng giọng nói. Tokenizer residual-VQ là mất mát; giọng nói hội thoại nghe máy móc so với các giọng cỡ ElevenLabs.
- Suy luận liên phương thức. Yêu cầu mô hình "hát về điều bạn thấy" vẫn thất bại thường xuyên hơn các tác vụ thuần thị giác.

Đây là những bài toán nghiên cứu mở. Qwen3-Omni (Bài 12.20) là nỗ lực mở tiên tiến nhất trong 2025.

```figure
any-to-any-stream
```

## Áp dụng

`code/main.py`:

- Định nghĩa phân bổ từ vựng bốn phương thức và in ra.
- Định tuyến một danh sách đầu vào đa phương thức (văn bản, hình ảnh, đoạn âm thanh, âm nhạc) qua bộ định tuyến tokenizer.
- Mô phỏng giải mã truyền trực tiếp cho một phản hồi text-to-speech (chuyển văn bản thành giọng nói) có đếm độ trễ.
- Tính thời gian tới byte âm thanh đầu tiên kỳ vọng với độ trễ của encoder, prefill và decoder.

## Đóng gói

Bài này tạo ra `outputs/skill-any-to-any-pipeline-auditor.md`. Cho một đặc tả sản phẩm hội thoại (phương thức vào, phương thức ra, mục tiêu độ trễ), nó kiểm toán các lựa chọn thiết kế họ MIO và tính ngân sách độ trễ.

## Bài tập

1. Sản phẩm của bạn nhận đầu vào giọng nói và trả đầu ra giọng nói. Mục tiêu ngân sách độ trễ đầu-cuối là bao nhiêu? Liệt kê các thành phần tiêu tốn thời gian.

2. SpeechTokenizer residual-VQ dùng 8 codebook. Hãy lập luận vì sao cần giải mã song song các tầng residual (thay vì tuần tự) và nó tiết kiệm bao nhiêu độ trễ.

3. Từ vựng của bạn có 32k văn bản + 4k hình ảnh + 4k giọng nói. Thêm 8k âm nhạc và ~10 dấu phân cách. Chi phí tham số ma trận embedding ở hidden dim 4096 là bao nhiêu?

4. Chain-of-visual-thought phát ra một hình ảnh trung gian. Những loại câu hỏi nào được lợi? Loại nào bị hại vì các token thêm vào?

5. Đọc Moshi (arXiv:2410.00037). Mô tả kỹ thuật "độc thoại nội tâm" của nó và so sánh với chain-of-visual-thought của MIO.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Any-to-any | "Đa phương thức vào/ra" | Một mô hình duy nhất nhận và phát văn bản, hình ảnh, giọng nói, âm nhạc theo mọi hướng |
| Residual-VQ | "Chồng tokenizer giọng nói" | Token hoá đa codebook trong đó mỗi tầng thêm thông tin; tầng cơ sở là nội dung, các tầng sau là ngữ điệu |
| SEED-Tokenizer | "Mã hình ảnh" | Tokenizer hình ảnh rời rạc với codebook 4096 mục mà MIO dùng |
| Chain-of-visual-thought | "Bảng nháp thị giác" | Mô hình sinh một hình ảnh trung gian như bước suy luận trước câu trả lời cuối |
| Time-to-first-audio-byte | "TTFAB" | Độ trễ từ giọng người dùng tới âm thanh đầu ra đầu tiên; <500ms để có cảm giác hội thoại |
| Four-stage curriculum | "Công thức huấn luyện" | Căn chỉnh -> xen kẽ -> tăng cường giọng nói -> SFT, theo đúng thứ tự đó |

## Đọc thêm

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)

---

[← Bài trước: Janus-Pro: Bộ mã hoá tách rời cho mô hình đa phương thức hợp nhất](../../15-janus-pro-decoupled-encoders/docs/vi.md) · [Bài sau: Mô hình video-ngôn ngữ (video-language): token thời gian và định vị theo thời gian →](../../17-video-language-temporal-grounding/docs/vi.md)
