# Vision Transformer (ViT)

> Cắt ảnh thành các patch, coi mỗi patch như một từ, chạy một transformer chuẩn. Không cần nhìn lại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 02 (Self-attention), Giai đoạn 4, Bài 04 (Phân loại ảnh)
**Time:** ~45 phút

## Mục tiêu học tập

- Tự viết patch embedding, positional embedding (embedding vị trí) học được, class token (token lớp) và các khối transformer encoder từ đầu để dựng một ViT (transformer thị giác) tối giản
- Giải thích vì sao người ta từng cho rằng ViT cần lượng dữ liệu pretraining khổng lồ, cho tới khi DeiT và MAE chứng minh điều ngược lại
- So sánh ViT, Swin và ConvNeXt theo prior kiến trúc của chúng (không có, attention cửa sổ cục bộ, conv backbone)
- Fine-tune một ViT đã pretrain trên bộ dữ liệu nhỏ bằng `timm` và công thức linear-probe / fine-tune chuẩn

## Vấn đề

Trong một thập kỷ, convolution đồng nghĩa với thị giác máy tính. CNN có inductive bias (thiên kiến quy nạp) mạnh — tính cục bộ, tính bất biến dịch — mà không ai nghĩ có thể thay thế. Rồi Dosovitskiy et al. (2020) cho thấy một transformer thuần áp lên các patch ảnh đã làm phẳng, không có bất kỳ cơ chế convolution nào, vẫn có thể ngang hoặc hơn CNN tốt nhất ở quy mô lớn.

Vấn đề nằm ở "quy mô lớn". ViT trên ImageNet-1k thua ResNet. ViT pretrain trên ImageNet-21k hoặc JFT-300M rồi fine-tune trên ImageNet-1k thì thắng. Kết luận là transformer thiếu prior hữu ích nhưng có thể học chúng từ đủ dữ liệu. Các công trình sau (DeiT, MAE, DINO) cho thấy với công thức huấn luyện đúng — tăng cường mạnh, pretraining tự giám sát, distillation — ViT cũng huấn luyện tốt trên dữ liệu nhỏ.

Tới 2026, CNN thuần vẫn cạnh tranh được trên thiết bị biên (ConvNeXt mạnh nhất), nhưng transformer chiếm ưu thế ở mọi chỗ khác: segmentation (Mask2Former, SegFormer), detection (DETR, RT-DETR), đa phương thức (CLIP, SigLIP), video (VideoMAE, VJEPA). Cấu trúc khối ViT là thứ bạn cần nắm.

## Khái niệm

### Pipeline

```mermaid
flowchart LR
    IMG["Ảnh<br/>(3, 224, 224)"] --> PATCH["Patch embedding<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["Làm phẳng thành<br/>(196, 768) token"]
    FLAT --> CAT["Chèn vào đầu<br/>[CLS] token"]
    CAT --> POS["Cộng positional<br/>embed đã học"]
    POS --> ENC["N khối<br/>transformer encoder"]
    ENC --> CLS["Lấy đầu ra [CLS]<br/>token output"]
    CLS --> HEAD["Bộ phân loại MLP"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

Bảy bước. Patch -> token -> attention -> bộ phân loại. Mọi biến thể (DeiT, Swin, ConvNeXt, pretraining MAE) chỉ đổi một hoặc hai trong bảy bước và để nguyên phần còn lại.

### Patch embedding

Conv đầu tiên là bí quyết. Kernel size 16, stride 16, nên ảnh 224x224 thành lưới 14x14 các patch 16x16, mỗi patch được chiếu thành embedding 768 chiều. Một conv duy nhất vừa chia patch vừa chiếu tuyến tính.

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196 patch = 196 token. Chiều đặc trưng của mỗi token là 768 (ViT-B), 1024 (ViT-L) hoặc 1280 (ViT-H).

### Token lớp

Một vector học được duy nhất được chèn vào đầu chuỗi:

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

Sau N khối transformer, đầu ra của `[CLS]` là biểu diễn ảnh toàn cục. Đầu phân loại chỉ đọc vector này.

### Embedding vị trí

Transformer không có khái niệm vị trí không gian sẵn có. Cộng một vector học được vào mỗi token:

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

Embedding này là một tham số của mô hình; huấn luyện dựa trên gradient sẽ thích nghi nó với cấu trúc ảnh 2D. Có các phương án sinusoidal 2D nhưng hiếm khi dùng trong thực tế.

### Khối transformer encoder

Chuẩn. Multi-head self-attention (tự chú ý), MLP, kết nối residual, pre-LayerNorm.

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 xếp 12 khối như vậy, mỗi khối 12 attention head, tổng cộng 86M tham số.

### Vì sao dùng pre-LN

Transformer thời đầu dùng post-LN (`x = LN(x + sublayer(x))`) và khó huấn luyện quá 6-8 lớp nếu không warmup. Pre-LN (`x = x + sublayer(LN(x))`) huấn luyện mạng sâu ổn định mà không cần warmup. Mọi ViT và mọi LLM hiện đại đều dùng pre-LN.

### Đánh đổi kích thước patch

- Patch 16x16 -> 196 token, chuẩn.
- Patch 32x32 -> 49 token, nhanh hơn nhưng độ phân giải thấp hơn.
- Patch 8x8 -> 784 token, mịn hơn nhưng chi phí attention O(n^2) tăng tệ.

Patch lớn hơn = ít token hơn = nhanh hơn nhưng ít chi tiết không gian hơn. SwinV2 dùng patch 4x4 trong các cửa sổ phân cấp.

### Công thức của DeiT để huấn luyện ViT trên ImageNet-1k

ViT gốc cần JFT-300M để hạ CNN. DeiT (Touvron et al., 2020) huấn luyện ViT-B đạt 81.8% top-1 chỉ trên ImageNet-1k với bốn thay đổi:

1. Tăng cường mạnh: RandAugment, Mixup, CutMix, Random Erasing.
2. Stochastic depth (bỏ ngẫu nhiên cả khối khi huấn luyện).
3. Repeated augmentation (cùng một ảnh được lấy 3 lần mỗi batch).
4. Distillation từ một teacher CNN (tuỳ chọn, nâng độ chính xác thêm).

Mọi công thức huấn luyện ViT hiện đại đều bắt nguồn từ DeiT.

### Swin vs ConvNeXt

- **Swin** (Liu et al., 2021) — attention dựa trên cửa sổ. Mỗi khối chỉ chú ý trong một cửa sổ cục bộ; các khối xen kẽ dịch cửa sổ để trộn thông tin giữa các cửa sổ. Mang lại prior cục bộ kiểu CNN trong khi vẫn giữ toán tử attention.
- **ConvNeXt** (Liu et al., 2022) — CNN được thiết kế lại, khớp các lựa chọn kiến trúc của Swin (depthwise conv, LayerNorm, GELU, inverted bottleneck). Cho thấy khoảng cách không phải "attention vs convolution" mà là "công thức huấn luyện hiện đại + kiến trúc".

Năm 2026, ConvNeXt-V2 và Swin-V2 đều đạt mức production; lựa chọn đúng phụ thuộc vào stack suy luận của bạn (ConvNeXt biên dịch tốt hơn cho thiết bị biên) và kho dữ liệu pretrain.

### Pretraining MAE

Masked Autoencoder (He et al., 2022): che ngẫu nhiên 75% patch, huấn luyện encoder chỉ xử lý 25% patch còn nhìn thấy, huấn luyện một decoder nhỏ để tái tạo các patch bị che từ đầu ra của encoder. Sau pretraining, bỏ decoder và fine-tune encoder.

MAE giúp ViT huấn luyện được chỉ với ImageNet-1k, đạt SOTA, và là công thức tự giám sát mặc định hiện nay.

```figure
batchnorm-inference
```

## Tự tay làm

### Bước 1: Patch embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

Một conv, một flatten, một transpose. Đó là toàn bộ bước ảnh-sang-token.

### Bước 2: Khối transformer

Pre-LN, multi-head self-attention, MLP với GELU, kết nối residual.

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` lo phần tách head, scaled dot-product và phép chiếu đầu ra. `batch_first=True` để shape là `(N, seq, dim)`.

### Bước 3: ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

Khoảng 2.8M tham số — một ViT nhỏ chạy được trên CPU. ViT-B thực tế là 86M; cùng định nghĩa lớp với `dim=768, depth=12, num_heads=12`.

### Bước 4: Kiểm tra nhanh — suy luận một ảnh

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

Phải chạy không lỗi. Xác suất cộng lại bằng 1.

## Áp dụng

`timm` có sẵn mọi biến thể ViT với trọng số pretrain ImageNet. Một dòng:

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` là lựa chọn mặc định cho production với vision transformer năm 2026. Hỗ trợ ViT, DeiT, Swin, Swin-V2, ConvNeXt, ConvNeXt-V2, MaxViT, MViT, EfficientFormer và hàng chục mô hình khác dưới cùng một API.

Với công việc đa phương thức (ảnh + văn bản), `transformers` có CLIP, SigLIP, BLIP-2, LLaVA. Encoder ảnh trong tất cả các mô hình đó đều là một biến thể của ViT.

## Đóng gói

Bài học này tạo ra:

- `outputs/prompt-vit-vs-cnn-picker.md` — một prompt chọn giữa ViT, ConvNeXt hoặc Swin dựa trên kích thước dữ liệu, compute và stack suy luận.
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — một skill kiểm tra shape của patch embedding và positional embedding trong ViT có khớp độ dài chuỗi mà mô hình mong đợi, giúp bắt lỗi port phổ biến nhất.

## Bài tập

1. **(Dễ)** In shape của mọi tensor trung gian trong một lượt forward qua ViT nhỏ ở trên. Xác nhận: input `(N, 3, 64, 64)` -> patch `(N, 16, 192)` -> với CLS `(N, 17, 192)` -> đầu vào bộ phân loại `(N, 192)` -> đầu ra `(N, num_classes)`.
2. **(Trung bình)** Fine-tune một ViT-S/16 đã pretrain từ `timm` trên bộ dữ liệu CIFAR tổng hợp ở Bài 4. So sánh với fine-tune ResNet-18 trên cùng dữ liệu. Báo cáo thời gian huấn luyện và độ chính xác cuối.
3. **(Khó)** Cài đặt pretraining MAE cho ViT nhỏ: che 75% patch, huấn luyện encoder + một decoder nhỏ để tái tạo các patch bị che. Đánh giá độ chính xác linear-probe trên dữ liệu tổng hợp trước và sau pretraining.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Patch embedding | "Conv đầu tiên" | Một conv có kernel size = stride = kích thước patch; biến ảnh thành lưới các embedding token |
| Class token | "[CLS]" | Một vector học được chèn vào đầu chuỗi token; đầu ra cuối của nó là biểu diễn ảnh toàn cục |
| Positional embedding | "Pos học được" | Một vector học được cộng vào mỗi token để transformer biết mỗi patch đến từ đâu |
| Pre-LN | "LayerNorm trước sublayer" | Biến thể transformer ổn định: `x + sublayer(LN(x))` thay vì `LN(x + sublayer(x))` |
| Multi-head attention | "Attention song song" | Attention transformer chuẩn được tách thành num_heads không gian con độc lập, rồi nối lại |
| ViT-B/16 | "Base, patch 16" | Kích thước chuẩn: dim=768, depth=12, heads=12, patch_size=16, image=224; ~86M tham số |
| DeiT | "ViT tiết kiệm dữ liệu" | ViT huấn luyện chỉ trên ImageNet-1k với tăng cường mạnh; chứng minh không nhất thiết phải có bộ dữ liệu pretrain lớn |
| MAE | "Masked autoencoder" | Pretraining tự giám sát: che 75% patch, tái tạo; công thức pretraining ViT thống trị |

## Đọc thêm

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — bài báo ViT
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — cách huấn luyện ViT chỉ trên ImageNet-1k
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — pretraining MAE
- [tài liệu timm](https://huggingface.co/docs/timm) — tài liệu tham chiếu cho mọi vision transformer bạn sẽ dùng trong production

---

[← Bài trước: Thị giác 3D](../../13-3d-vision-nerf/docs/vi.md) · [Bài sau: Thị giác thời gian thực →](../../15-real-time-edge/docs/vi.md)
