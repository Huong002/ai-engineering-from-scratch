# Emu3: Dự đoán token kế tiếp cho sinh ảnh và sinh video

> Emu3 của BAAI (Wang et al., tháng 9 năm 2024) là kết quả năm 2024 lẽ ra đã phải khép lại cuộc tranh luận khuếch tán so với autoregressive (tự hồi quy). Một transformer decoder-only kiểu Llama duy nhất, chỉ huấn luyện trên mục tiêu next-token prediction (dự đoán token kế tiếp), trên một từ vựng thống nhất gồm văn bản + token ảnh VQ + token video VQ 3D, đánh bại SDXL về sinh ảnh (image generation) và LLaVA-1.6 về nhận thức. Không có loss CLIP. Không có lịch khuếch tán. Classifier-free guidance được dùng lúc suy luận để cải thiện chất lượng, nhưng mục tiêu huấn luyện cốt lõi vẫn là dự đoán token kế tiếp với teacher forcing. Công bố trên Nature. Bài này đọc luận đề của Emu3 — vì sao một tokenizer tốt hơn cộng với quy mô là tất cả những gì bạn cần — và đối chiếu với các hướng tiếp cận khuếch tán.

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Giai đoạn 12, Bài 11 (Chameleon)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích vì sao mục tiêu next-token một loss duy nhất của Emu3 vẫn hiệu quả, bất chấp giả định tồn tại lâu nay rằng cần khuếch tán để có chất lượng ảnh tốt.
- Mô tả tokenizer video 3D: một codebook VQ không gian–thời gian trông ra sao, vì sao các patch trải theo thời gian.
- So sánh Emu3 với Stable Diffusion XL trên (chi phí huấn luyện, chi phí suy luận, trần chất lượng).
- Kể tên ba vai trò của cùng một mô hình Emu3: Emu3-Gen (sinh ảnh), Emu3-Chat (nhận thức), Emu3-Stage2 (sinh video).

## Vấn đề

Hiểu biết thông thường suốt năm 2024: sinh ảnh cần khuếch tán. Lập luận: token ảnh rời rạc làm mất quá nhiều thông tin để tái tạo chi tiết, và lấy mẫu tự hồi quy tích lũy sai số qua hàng nghìn token. Stable Diffusion, DALL-E 3, Imagen, Midjourney đều dùng một dạng khuếch tán nào đó. Chameleon (Bài 12.11) đã bác bỏ một phần điều này ở quy mô nhỏ nhưng chưa đạt chất lượng ngang SDXL.

Emu3 tấn công trực diện lập luận đó. Tuyên bố: visual tokenizer (bộ token hóa thị giác) tốt hơn + đủ quy mô + next-token loss = sinh ảnh vượt khuếch tán, trong cùng một mô hình cũng làm được nhận thức.

Canh bạc này gây tranh cãi khi mới công bố. Hai năm sau, họ mô hình sinh hợp nhất mã nguồn mở (Emu3, Show-o, Janus-Pro, Transfusion) là hướng mặc định cho nghiên cứu; các mô hình biên trong sản xuất dường như dùng một biến thể nào đó.

## Khái niệm

### Tokenizer của Emu3

Thành phần then chốt là visual tokenizer. Emu3 huấn luyện một tokenizer riêng thuộc lớp IBQ (Inverse Bottleneck Quantizer, họ SBER-MoVQGAN) với mức giảm độ phân giải 8x8 cho mỗi token. Một ảnh 512x512 trở thành 64x64 = 4096 token với codebook kích thước 32768.

Con số này lớn hơn 1024 token cho mỗi ảnh 512x512 của Chameleon ở K=8192 nhưng rẻ hơn trên mỗi token (tra codebook nhỏ hơn, codec đơn giản hơn). Chỉ số then chốt: PSNR tái tạo đạt 30.5 dB, ngang ngửa không gian latent (biểu diễn ẩn) liên tục của Stable Diffusion ở 32 dB.

Với video: một tokenizer VQ 3D mã hóa một patch không gian–thời gian (4x4x4 pixel) thành một số nguyên. Một clip 4 giây ở 8 FPS có 32 khung hình; ở 256x256 với mức giảm 4 lần theo không gian và 4 lần theo thời gian, số token là (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 token.

Chất lượng tokenizer chính là trần. Đóng góp của Emu3 một phần nằm ở câu "chúng tôi đã huấn luyện một tokenizer rất tốt".

### Huấn luyện với một loss duy nhất

Emu3 dùng một mục tiêu duy nhất: dự đoán token kế tiếp trên một từ vựng dùng chung cho token văn bản, token ảnh 2D và token video 3D. Trọng số được nhân với các hệ số riêng theo phương thức trong lúc huấn luyện để cân bằng đóng góp, nhưng hàm loss thì giống hệt nhau.

Huấn luyện trên một hỗn hợp gồm:
- Sinh ảnh: `<text caption> <image> image_tokens </image>`
- Nhận thức ảnh: `<image> image_tokens </image> <question> text_tokens`
- Sinh video: `<text caption> <video> video_tokens </video>`
- Nhận thức video: tương tự.
- Chỉ văn bản: NTP tiêu chuẩn.

Mô hình học khi nào phát ra token ảnh và khi nào phát token văn bản từ phân phối dữ liệu. Việc sinh nội dung nảy sinh khi mô hình dự đoán token ảnh sau thẻ `<image>`.

### Classifier-free guidance và temperature

Sinh ảnh tự hồi quy tốt lên rất nhiều nhờ classifier-free guidance (CFG) lúc suy luận. Emu3 dùng nó: sinh hai lần, một lần với caption đầy đủ, một lần với caption rỗng, rồi trộn các logit với một trọng số guidance (thường 3.0-7.0). Đây đúng là mẹo CFG mà khuếch tán dùng, được vay sang bối cảnh tự hồi quy.

Temperature quan trọng: quá cao thì sinh artifact; quá thấp thì mode collapse. Emu3 khuyến nghị temperature 1.0 cho nhận thức, 0.8 cho sinh ảnh.

### Ba vai trò, một mô hình

Emu3 được phát hành dưới dạng ba API khác nhau về chức năng nhưng chỉ một bộ trọng số nền:

- Emu3-Gen. Sinh ảnh. Đầu vào văn bản, đầu ra token ảnh.
- Emu3-Chat. VQA và tạo caption. Đầu vào ảnh (token), đầu ra văn bản.
- Emu3-Stage2. Sinh video và VQA video. Đầu vào văn bản hoặc video, đầu ra văn bản hoặc video.

Không có head riêng cho từng nhiệm vụ. Chỉ là các prompt template khác nhau. Cùng một checkpoint.

### Kết quả đánh giá

Từ bài báo Emu3 (tháng 9 năm 2024):

- Sinh ảnh: thắng SDXL trên FID của MJHQ-30K (5.4 so với 5.6), tổng thể GenEval (0.54 so với 0.55 — ngang nhau về mặt thống kê), và điểm tổng hợp của Deep-Eval ngang bằng.
- Nhận thức ảnh: thắng LLaVA-1.6 trên VQAv2 (75.1 so với 72.4) và gần ngang trên MMMU.
- Sinh video: chất lượng clip 4 giây với FVD cạnh tranh so với các mô hình thời Sora được đánh giá công khai.

Các con số không phải lúc nào cũng thắng — Emu3 đổi một điểm chỗ này lấy một điểm chỗ kia — nhưng tuyên bố "next-token prediction là tất cả những gì bạn cần" vẫn đứng vững trên nhiều phương thức.

### Chi phí tính toán

Emu3 được huấn luyện trên ~300 tỷ token đa phương thức với một mô hình 7B tham số. Số giờ GPU xấp xỉ với tiền huấn luyện Llama-2-7B (2k-4k GPU-năm trên silicon hạng A100). Các mô hình khuếch tán như Stable Diffusion 3 huấn luyện với ngân sách tương tự nhưng cần text encoder riêng và pipeline phức tạp hơn.

Lúc suy luận, Emu3 chậm hơn SDXL trên mỗi ảnh: 4096 token ảnh ở 30 tok/s là khoảng 2 phút cho một ảnh 512x512, so với 2-5 giây của SDXL. Speculative decoding và tối ưu KV-cache thu hẹp khoảng cách nhưng không xóa bỏ nó. Sinh ảnh tự hồi quy ngốn tính toán; đây là đánh đổi thường trực.

### Vì sao điều này quan trọng

Đóng góp sâu xa của Emu3 là về mặt khái niệm. Nếu dự đoán token kế tiếp mở rộng được để ngang khuếch tán về sinh ảnh, thì hướng mô hình hợp nhất (một loss, một backbone, mọi phương thức) là khả thi. Các mô hình tương lai không cần text encoder riêng, bộ lập lịch khuếch tán riêng, VAE riêng. Một transformer, mỗi phương thức một tokenizer, và quy mô.

Show-o, Janus-Pro và InternVL-U đều xây dựng trên hoặc thách thức luận đề này. Các phòng thí nghiệm Trung Quốc (BAAI, DeepSeek) công bố mạnh mẽ theo hướng này hơn các phòng thí nghiệm Mỹ trong suốt 2025.

```figure
l5-emu3-next-token
```

## Áp dụng

`code/main.py` dựng hai phần đồ chơi:

- Một bộ tính số token của tokenizer VQ 2D và 3D: cho (resolution, patch, clip_length, FPS), tính số token cho ảnh và cho video.
- Một bộ lấy mẫu token ảnh tự hồi quy với classifier-free guidance ở một temperature.

Cách cài đặt CFG khớp công thức của Emu3 — trộn logit có điều kiện và không điều kiện với một trọng số guidance.

## Đóng gói

Bài này tạo ra `outputs/skill-token-gen-cost-analyzer.md`. Cho một spec sản phẩm sinh nội dung (ảnh hoặc video, độ phân giải mục tiêu, mức chất lượng, ngân sách độ trễ), nó tính số token, chi phí suy luận và chọn họ Emu3 hay khuếch tán.

## Bài tập

1. Emu3 tạo 4096 token cho mỗi ảnh 512x512 với mức giảm 8x8. Tính tương đương cho 1024x1024 và 2048x2048. Độ trễ suy luận sẽ ra sao?

2. Đọc Mục 3.3 của Emu3 về tokenizer video. Mô tả hình dạng patch VQ 3D và vì sao là 4x4x4 mà không phải 8x8x1.

3. Trọng số classifier-free guidance 5.0 so với 3.0: hiệu ứng thị giác là gì? Lần theo phép toán trong `code/main.py`.

4. Tính FLOPs huấn luyện cho Emu3-7B với 300 tỷ token và so sánh với Stable Diffusion 3. Mô hình nào tốn kém hơn để huấn luyện?

5. Emu3 thắng SDXL về FID nhưng không thắng trên VQAv2 so với các VLM chuyên dụng. Giải thích vì sao hướng tiếp cận một loss hợp nhất lại mạnh ở mặt này yếu ở mặt kia so với mô hình chuyên dụng trên các benchmark khác nhau.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | Loss tự hồi quy tiêu chuẩn: dự đoán token[i+1] khi biết token[0..i]; dùng được cho mọi phương thức một khi đã token hóa |
| IBQ tokenizer | "Inverse bottleneck quantizer" | Một lớp VQ-VAE có codebook lớn hơn (32768+) và tái tạo tốt hơn của Chameleon |
| 3D VQ | "Spatiotemporal quantizer" | Codebook được đánh chỉ số theo (thời gian, hàng, cột); một token phủ một khối 4x4x4 pixel |
| Classifier-free guidance | "CFG" | Trộn logit có điều kiện và không điều kiện với trọng số gamma; tăng chất lượng ảnh lúc suy luận |
| Unified vocabulary | "Shared tokens" | Văn bản + ảnh + video cùng lấy từ một không gian số nguyên; mô hình dự đoán phương thức nào xuất hiện tiếp theo |
| MJHQ-30K | "Image gen benchmark" | Benchmark chất lượng kiểu Midjourney với 30k prompt; Emu3 báo cáo FID trên đó |

## Đọc thêm

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)

---

[← Bài trước: Chameleon và các mô hình đa phương thức chỉ dùng token với early fusion (hợp nhất sớm)](../../11-chameleon-early-fusion-tokens/docs/vi.md) · [Bài sau: Transfusion: Văn bản autoregressive (tự hồi quy) + Ảnh diffusion (khuếch tán) trong một Transformer →](../../13-transfusion-autoregressive-diffusion/docs/vi.md)
