# Inpainting, Outpainting & Chỉnh sửa ảnh

> Text-to-image tạo ra thứ mới. Inpainting (vẽ bù) sửa những thứ cũ. Trong sản xuất thực tế, 70% công việc ảnh có thu phí là chỉnh sửa — đổi nền, xoá logo, mở rộng khung hình, vẽ lại một bàn tay. Inpainting chính là chỗ diffusion chứng minh giá trị của nó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 07 (Latent diffusion), Giai đoạn 8, Bài 08 (ControlNet & LoRA)
**Time:** ~75 phút

## Vấn đề

Một khách hàng gửi cho bạn tấm ảnh sản phẩm hoàn hảo nhưng có một tấm biển gây nhiễu trong nền. Bạn muốn xoá tấm biển và giữ mọi thứ còn lại giống hệt từng pixel. Bạn không thể chạy text-to-image từ đầu — kết quả sẽ khác màu, khác ánh sáng, khác góc chụp sản phẩm. Bạn muốn tái tạo *chỉ* vùng mask (mặt nạ), và muốn lần tái tạo đó tôn trọng context (ngữ cảnh) xung quanh.

Đó là inpainting. Các biến thể:

- **Inpainting.** Tái tạo bên trong mask, giữ nguyên pixel bên ngoài.
- **Outpainting (vẽ mở rộng).** Tái tạo bên ngoài mask (hoặc vượt ra ngoài khung hình), giữ nguyên bên trong.
- **Chỉnh sửa ảnh.** Tái tạo toàn bộ ảnh nhưng vẫn giữ độ trung thực về ngữ nghĩa hoặc cấu trúc so với bản gốc (SDEdit, InstructPix2Pix).

Mọi pipeline diffusion năm 2026 đều có chế độ inpainting. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Tất cả đều hoạt động theo cùng một nguyên lý.

## Khái niệm

![Inpainting: khử nhiễu có nhận biết mask với việc tiêm lại bảo toàn ngữ cảnh](../assets/inpainting.svg)

### Cách làm ngây thơ (và tại sao nó sai)

Chạy text-to-image tiêu chuẩn với một mask. Ở mỗi bước lấy mẫu, thay vùng không bị mask của latent (ẩn) đang nhiễu bằng ảnh sạch đã được khuếch tán tiến. Nó chạy được... nhưng tệ. Artifact ở biên lan vào do hiện tượng blending (hoà trộn), vì mô hình không có thông tin gì về nội dung nằm trong vùng mask.

### Mô hình inpainting đúng chuẩn

Huấn luyện một U-Net sửa đổi nhận 9 kênh đầu vào thay vì 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Các kênh thêm là bản sao ảnh nguồn đã được VAE mã hoá, cộng thêm một mask một kênh. Khi huấn luyện, bạn mask ngẫu nhiên các vùng của ảnh và dạy mô hình chỉ khử nhiễu vùng bị mask, còn vùng không bị mask được đưa vào như tín hiệu điều kiện sạch. Khi suy luận, mô hình có thể "nhìn thấy" thứ bao quanh vùng mask và tạo ra phần bù mạch lạc.

SD-Inpaint, SDXL-Inpaint, Flux-Fill đều dùng đầu vào 9 kênh này (hoặc tương đương). Trong diffusers là `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — chỉnh sửa miễn phí

Thêm nhiễu vào ảnh nguồn tới một mốc trung gian `t`, rồi chạy chuỗi ngược từ `t` về 0 với prompt mới. Không cần huấn luyện lại. Việc chọn `t` khởi đầu đánh đổi độ trung thực lấy tự do sáng tạo:

- `t/T = 0.3` → gần như giống hệt nguồn, thay đổi phong cách nhỏ
- `t/T = 0.6` → chỉnh sửa vừa phải, giữ cấu trúc thô
- `t/T = 0.9` → sinh gần như từ nhiễu, gần như không giữ được nguồn

### InstructPix2Pix (Brooks et al., 2023)

Fine-tune một mô hình diffusion trên các bộ ba `(input_image, instruction, output_image)`. Khi suy luận, điều kiện hoá trên cả ảnh đầu vào và một câu chỉ dẫn văn bản ("make it sunset", "add a dragon"). Hai thang CFG (guidance scale, tức hệ số hướng dẫn): thang cho ảnh và thang cho văn bản.

### RePaint (Lugmayr et al., 2022)

Giữ một mô hình diffusion vô điều kiện tiêu chuẩn. Ở mỗi bước ngược, lấy mẫu lại — thỉnh thoảng nhảy về trạng thái nhiễu hơn rồi tái tạo. Cách này tránh được artifact ở biên. Dùng khi bạn không có mô hình inpainting đã huấn luyện.

```figure
inpaint-mask-reinject
```

## Tự tay làm

`code/main.py` cài đặt một sơ đồ inpainting 1-D đồ chơi trên dữ liệu 5 chiều. Chúng ta huấn luyện một DDPM trên dữ liệu hỗn hợp 5-D, trong đó mỗi mẫu là 5 số thực lấy từ một trong hai cụm. Khi suy luận, chúng ta "mask" 2 trong 5 chiều, tiêm bản đã thêm nhiễu (khuếch tán tiến) của ba chiều không bị mask ở mỗi bước, và chỉ tái tạo các chiều bị mask.

### Bước 1: dữ liệu DDPM 5-D

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Bước 2: huấn luyện denoiser trên cả 5 chiều

DDPM tiêu chuẩn. Mạng xuất dự đoán nhiễu 5 chiều cho đầu vào nhiễu 5 chiều.

### Bước 3: khi suy luận, chạy ngược có nhận biết mask

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

Đây là cách làm ngây thơ và nó chạy được trên dữ liệu 1-D đồ chơi. Inpainting ảnh thật dùng đầu vào 9 kênh vì độ mạch lạc của texture quan trọng hơn.

### Bước 4: outpainting

Outpainting là inpainting với mask đảo ngược: mask phần khung hình mới (trước đó không tồn tại), lấp phần còn lại bằng ảnh gốc. Mục tiêu huấn luyện giống hệt.

## Cạm bẫy

- **Đường nối.** Cách làm ngây thơ để lại ranh giới nhìn thấy được vì thông tin gradient không chảy qua mask. Cách sửa: giãn mask thêm 8-16 pixel, hoặc dùng mô hình inpainting đúng chuẩn.
- **Rò rỉ mask.** Nếu vùng không bị mask của ảnh điều kiện có chất lượng thấp hoặc nhiễu, nó sẽ làm bẩn phần sinh ra bên trong mask. Hãy khử nhiễu hoặc làm mờ nhẹ.
- **CFG tương tác với kích thước mask.** CFG cao trên mask nhỏ = mảng bị bão hoà. Giảm CFG cho các chỉnh sửa nhỏ.
- **Vách trung thực của SDEdit.** Đi từ `t/T = 0.5` lên `t/T = 0.6` có thể làm mất danh tính của chủ thể. Hãy quét tham số và đặt checkpoint.
- **Prompt lệch.** Prompt nên mô tả *toàn bộ* ảnh, không chỉ nội dung mới. "A cat sitting on a chair" chứ không phải "a cat".

## Áp dụng

| Tác vụ | Pipeline |
|------|----------|
| Xoá vật thể, mask nhỏ | SD-Inpaint hoặc Flux-Fill, prompt tiêu chuẩn |
| Thay bầu trời | SD-Inpaint + "blue sky at sunset" |
| Mở rộng khung hình | Chế độ outpaint của SDXL (feather 8px) hoặc Flux-Fill với mask outpaint |
| Vẽ lại bàn tay / khuôn mặt | SD-Inpaint với prompt mô tả lại chủ thể + ControlNet-Openpose |
| Đổi phong cách một vùng | SDEdit ở `t/T=0.5` trên vùng bị mask |
| "Làm cho thành hoàng hôn" | InstructPix2Pix hoặc Flux-Kontext |
| Thay nền | SAM mask → SD-Inpaint |
| Độ trung thực cực cao | Flux-Fill hoặc GPT-Image (hosted) cho các ca khó nhất |

SAM (Segment Anything của Meta, 2023) + diffusion inpaint là pipeline xoá nền của năm 2026. SAM 2 (2024) chạy được trên video.

## Đóng gói

Lưu `outputs/skill-editing-pipeline.md`. Skill nhận một ảnh gốc + mô tả chỉnh sửa + mask tuỳ chọn (hoặc prompt cho SAM) và xuất ra: cách tạo mask, mô hình nền, các thang CFG (ảnh + văn bản), SDEdit-t hoặc chế độ inpainting, và checklist QA.

## Bài tập

1. **Dễ.** Trong `code/main.py`, thay đổi tỉ lệ số chiều bị mask từ 0.2 đến 0.8. Ở tỉ lệ nào thì chất lượng inpaint (phần dư ở các chiều bị mask) bằng với sinh vô điều kiện?
2. **Trung bình.** Cài đặt RePaint: cứ mỗi 10 bước ngược, nhảy lùi 5 bước (thêm nhiễu) rồi khử nhiễu lại. Đo xem nó có làm giảm phần dư ở biên mask hay không.
3. **Khó.** Dùng Hugging Face diffusers để so sánh: SD 1.5 Inpaint + ControlNet-Openpose với Flux.1-Fill trên 20 tác vụ vẽ lại khuôn mặt. Chấm điểm riêng mức bám tư thế và mức giữ danh tính.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Inpainting | "Lấp cái lỗ" | Tái tạo bên trong mask; giữ nguyên pixel bên ngoài. |
| Outpainting | "Mở rộng khung hình" | Tái tạo bên ngoài khung hình; giữ nguyên bên trong. |
| 9-channel U-Net | "Mô hình inpainting đúng chuẩn" | U-Net với đầu vào `noisy \| encoded-source \| mask`. |
| SDEdit | "Img2img kèm mức nhiễu" | Thêm nhiễu tới mốc `t`, khử nhiễu với prompt mới. |
| InstructPix2Pix | "Chỉnh sửa chỉ bằng văn bản" | Diffusion fine-tune trên các bộ ba (ảnh, chỉ dẫn, kết quả). |
| RePaint | "Không cần huấn luyện lại" | Thêm nhiễu lại định kỳ trong quá trình ngược để giảm đường nối. |
| SAM | "Segment Anything" | Bộ sinh mask bằng cú nhấp hoặc khung; đi cặp với inpaint. |
| Flux-Kontext | "Chỉnh sửa kèm ngữ cảnh" | Biến thể Flux nhận ảnh tham chiếu + chỉ dẫn để chỉnh sửa. |

## Ghi chú sản xuất: pipeline chỉnh sửa rất nhạy với độ trễ

Người dùng chỉnh sửa ảnh kỳ vọng vòng phản hồi dưới 5 giây. Một lượt SDXL-Inpaint 30 bước ở 1024² mất 3-4 s trên L4, cộng thêm tạo mask SAM (~200 ms) và encode/decode VAE (~500 ms tổng). Xét theo góc độ phục vụ (serving), nút thắt ở đây là TTFT chứ không phải throughput — batch 1, độ đồng thời thấp, tối ưu từng khâu:

- **SAM-H là khâu chậm.** SAM-H ở 1024² mất ~200 ms; SAM-ViT-B mất ~40 ms với mất mát chất lượng nhỏ. SAM 2 (video) thêm chi phí thời gian; đừng dùng nó cho việc chỉnh sửa một ảnh đơn.
- **Bỏ qua bước encode khi có thể.** `pipe.image_processor.preprocess(img)` mã hoá thành latent. Nếu bạn đã có latent từ lượt sinh trước (thường gặp trong các UI chỉnh sửa lặp), hãy truyền thẳng qua `latents=...` để bỏ một lần encode VAE.
- **Giãn mask cũng ảnh hưởng throughput.** Mask nhỏ nghĩa là phần lớn lượt forward của U-Net bị lãng phí (pixel không bị mask dù sao cũng bị ép giá trị). `StableDiffusionInpaintPipeline` của `diffusers` vẫn chạy toàn bộ U-Net; chỉ các biến thể 9 kênh đúng chuẩn mới tận dụng được phần tính toán bị mask.
- **Flux-Kontext là câu trả lời của năm 2025.** Một lượt forward duy nhất trên `(source_image, instruction)` — không cần mask riêng, không cần quét nhiễu SDEdit. Trên H100 nó cho ra một bản chỉnh sửa trong ~1.5 s. Bài học về kiến trúc: gộp các khâu lại.

## Đọc thêm

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — inpainting không cần huấn luyện.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — chỉnh sửa bằng chỉ dẫn văn bản.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, nguồn tạo mask.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — SAM cho video.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — chỉnh sửa ở mức attention.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — công cụ năm 2024.

---

[← Bài trước: ControlNet, LoRA & điều kiện hoá](../../08-controlnet-lora-conditioning/docs/vi.md) · [Bài sau: Sinh video (Video Generation) →](../../10-video-generation/docs/vi.md)
