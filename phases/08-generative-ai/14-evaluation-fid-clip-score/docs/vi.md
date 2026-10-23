# Đánh giá — FID, CLIP score, sở thích con người

> Mọi bảng xếp hạng mô hình sinh đều dẫn FID, CLIP score và một tỉ lệ thắng từ đấu trường sở thích con người. Mỗi con số đều có một dạng lỗi mà một nhà nghiên cứu kiên trì có thể khai thác. Nếu bạn không biết các dạng lỗi đó, bạn không phân biệt được một cải thiện thật với một màn chơi điểm.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 8, Bài 01 (Phân loại), Giai đoạn 2, Bài 04 (Độ đo đánh giá)
**Time:** ~45 phút

## Vấn đề

Một mô hình sinh được đánh giá trên *sample quality* (chất lượng mẫu) và *mức bám điều kiện*. Cả hai đều không có độ đo dạng đóng. Mô hình của bạn phải kết xuất 10.000 ảnh; phải có thứ gì đó gán số cho chúng; bạn phải tin các con số đó xuyên qua nhiều họ mô hình, nhiều độ phân giải, nhiều kiến trúc. Ba độ đo đã sống sót qua thử thách 2014-2026:

- **FID (Fréchet inception distance — khoảng cách Fréchet Inception).** Khoảng cách giữa hai distribution (phân phối) — thật và sinh ra — trong không gian đặc trưng của mạng Inception. Thấp hơn là tốt hơn.
- **CLIP score (điểm CLIP).** Độ tương đồng cosine giữa embedding CLIP-image của một ảnh sinh ra và embedding CLIP-text của một prompt. Cao hơn là tốt hơn. Đo mức bám prompt.
- **Sở thích con người.** Cho hai mô hình đối đầu trực tiếp trên cùng một prompt, để con người (hoặc một mô hình cỡ GPT-4) chọn cái tốt hơn, rồi tổng hợp thành điểm Elo.

Bạn cũng sẽ thấy: IS — inception score (điểm inception), phần lớn đã bị loại bỏ — KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Mỗi cái sửa một lỗi của cái trước.

## Khái niệm

![FID, CLIP và sở thích: ba trục, các dạng lỗi khác nhau](../assets/evaluation.svg)

### FID — chất lượng mẫu

Heusel và cộng sự (2017). Các bước:

1. Trích đặc trưng Inception-v3 (2048 chiều) cho N ảnh thật và N ảnh sinh ra.
2. Khớp một phân phối Gaussian cho mỗi tập: tính mean `μ_r, μ_g` và covariance `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Diễn giải: khoảng cách Fréchet giữa hai Gaussian đa biến trong không gian đặc trưng. Thấp hơn = các phân phối giống nhau hơn.

Các dạng lỗi:
- **Thiên lệch khi N nhỏ.** FID là trung bình bình phương trên phân phối đặc trưng — N nhỏ đánh giá thấp covariance, cho FID thấp một cách giả tạo. Luôn dùng N ≥ 10.000.
- **Phụ thuộc Inception.** Inception-v3 được huấn luyện trên ImageNet. Những miền xa ImageNet (khuôn mặt, nghệ thuật, ảnh chữ) cho FID vô nghĩa. Hãy dùng feature extractor (bộ trích đặc trưng) riêng cho miền.
- **Chơi điểm.** Overfitting vào prior của Inception cho FID thấp mà chất lượng thị giác không cải thiện. Khắc phục bằng CMMD (bên dưới).

### CLIP score — mức bám prompt

Radford và cộng sự (2021). Với một ảnh sinh ra + prompt:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Lấy trung bình trên 30k ảnh sinh ra → một số vô hướng so sánh được giữa các mô hình.

Các dạng lỗi:
- **Điểm mù của chính CLIP.** CLIP suy luận thành phần yếu ("a red cube on a blue sphere" thường thất bại). Các mô hình có thể xếp hạng cao trên CLIP score mà không thực sự bám theo prompt phức tạp.
- **Thiên lệch về prompt ngắn.** Prompt ngắn có nhiều kết quả khớp CLIP-image hơn trong thực tế. Prompt dài có CLIP score thấp hơn một cách máy móc.
- **Chơi prompt.** Chèn "high quality, 4k, masterpiece" vào prompt làm phồng CLIP score mà không cải thiện gắn kết ảnh-chữ.

CMMD (Jayasumana và cộng sự, 2024) sửa một số điểm này: dùng đặc trưng CLIP thay cho Inception, maximum-mean discrepancy thay cho Fréchet. Tốt hơn trong việc phát hiện những khác biệt chất lượng tinh tế.

### Sở thích con người — chân lý gốc

Chọn một tập prompt. Sinh bằng mô hình A và mô hình B. Cho con người (hoặc một LLM giám khảo mạnh) xem từng cặp. Tổng hợp số trận thắng thành điểm Elo hoặc Bradley-Terry. Các benchmark:

- **PartiPrompts (Google)**: 1.600 prompt đa dạng (diversity) trên 12 hạng mục.
- **HPSv2**: 107k chú thích của con người, được dùng rộng rãi làm đại diện tự động.
- **ImageReward**: 137k cặp sở thích prompt-ảnh, giấy phép MIT.
- **PickScore**: huấn luyện trên 2,6M sở thích từ Pick-a-Pic.
- **Đấu trường ảnh kiểu Chatbot-Arena**: https://imagearena.ai/ và những nơi khác.

Các dạng lỗi:
- **Phương sai giám khảo.** Người không chuyên có sở thích khác chuyên gia. Dùng cả hai.
- **Phân phối prompt.** Prompt được chọn lọc kỹ sẽ có lợi cho một họ mô hình. Luôn ghi lại.
- **Giám khảo LLM bị hack phần thưởng.** Giám khảo GPT-4 bị lừa bởi đầu ra đẹp nhưng sai. Hãy tam giác hoá với con người.

## Dùng kết hợp

Một báo cáo eval cho production nên gồm:

1. FID trên 10-30k mẫu so với một phân phối thật giữ lại (chất lượng mẫu).
2. CLIP score / CMMD trên cùng số mẫu so với prompt của chúng (mức bám).
3. Tỉ lệ thắng trong một đấu trường mù so với mô hình trước đó (sở thích tổng thể).
4. Phân tích dạng lỗi: 50 đầu ra lấy ngẫu nhiên, gắn cờ những vấn đề đã biết (giải phẫu bàn tay, kết xuất chữ, số lượng vật thể nhất quán).

Bất kỳ độ đo đơn lẻ nào cũng là một lời nói dối. Ba độ đo củng cố lẫn nhau + đánh giá định tính mới là một tuyên bố.

```figure
gx-fid-distributions
```

## Tự tay làm

`code/main.py` triển khai FID, CLIP-score-like, và tổng hợp Elo trên các "vector đặc trưng" tổng hợp (chúng ta dùng vector 4 chiều thay cho đặc trưng Inception). Bạn sẽ thấy:

- Tính FID trên N nhỏ và N lớn — độ thiên lệch.
- "CLIP score" như độ tương đồng cosine giữa các tập đặc trưng.
- Quy tắc cập nhật Elo từ một luồng sở thích tổng hợp.

### Bước 1: FID trong bốn dòng

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Bước 2: Độ tương đồng cosine kiểu CLIP

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Bước 3: Tổng hợp Elo

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Cạm bẫy

- **FID ở N=1000.** Công thức heuristic không đáng tin dưới N=10k. Các bài báo báo cáo FID với N nhỏ là đang chơi điểm.
- **So sánh FID giữa các độ phân giải.** Việc resize 299×299 của Inception làm thay đổi phân phối đặc trưng. Chỉ so sánh ở cùng độ phân giải.
- **Chỉ báo cáo một seed.** Chạy tối thiểu 3 seed. Báo cáo độ lệch chuẩn.
- **CLIP score bị thổi phồng nhờ prompt phủ định.** Một số pipeline tăng CLIP bằng cách overfit prompt. Kiểm tra hiện tượng bão hoà thị giác.
- **Elo thiên lệch do trùng prompt.** Nếu cả hai mô hình đã thấy một prompt benchmark trong lúc huấn luyện, Elo là vô nghĩa. Dùng tập prompt giữ lại.
- **Đánh giá con người bị lệch do đám đông trả tiền.** Người gán nhãn trên Prolific, MTurk trẻ hơn và thiên về công nghệ. Trộn với chuyên gia nghệ thuật/thiết kế được tuyển chọn.

## Áp dụng

Quy trình eval production năm 2026:

| Trụ cột | Tối thiểu | Đề xuất |
|--------|---------|-------------|
| Chất lượng mẫu | FID trên 10k so với tập thật giữ lại | + CMMD trên 5k + FID trên tập con theo từng hạng mục |
| Bám prompt | CLIP score trên 30k | + HPSv2 + ImageReward + hỏi đáp kiểu VQA |
| Sở thích | 200 cặp mù so với baseline | + 2000 cặp có con người + giám khảo LLM + Chatbot Arena |
| Phân tích lỗi | 50 do người gắn cờ | 500 do người gắn cờ + bộ phân loại an toàn tự động |

Cả bốn trụ cột trong một báo cáo = một tuyên bố. Bất kỳ cái nào đứng một mình = marketing.

## Đóng gói

Lưu `outputs/skill-eval-report.md`. Skill nhận một checkpoint mô hình mới + baseline và xuất ra một kế hoạch eval đầy đủ: kích thước mẫu, độ đo, đầu dò dạng lỗi, tiêu chí phê duyệt.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. So sánh FID ở N=100 và N=1000 trên cùng các phân phối tổng hợp. Báo cáo độ lớn thiên lệch.
2. **Trung bình.** Cài đặt CMMD từ các đặc trưng kiểu CLIP tổng hợp (xem Jayasumana và cộng sự, 2024 để có công thức). So sánh độ nhạy với các khác biệt chất lượng so với FID.
3. **Khó.** Tái lập thiết lập HPSv2: lấy 1000 cặp ảnh-prompt từ một tập con của Pick-a-Pic, fine-tune một bộ chấm điểm nhỏ dựa trên CLIP trên các sở thích đó, rồi đo mức đồng ý của nó với một tập giữ lại.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | Khoảng cách Fréchet giữa các phân phối Gaussian khớp vào đặc trưng Inception của ảnh thật và ảnh sinh. |
| CLIP score | "Độ tương đồng chữ-ảnh" | Độ tương đồng cosine giữa embedding ảnh và embedding chữ của CLIP. |
| CMMD | "Vật thay thế cho FID" | MMD trên đặc trưng CLIP; ít thiên lệch hơn, không giả định Gaussian. |
| IS | "Điểm inception" | Exp KL(p(y|x) || p(y)); tương quan kém trên các mô hình hiện đại, đã bị loại bỏ. |
| HPSv2 / ImageReward / PickScore | "Đại diện sở thích đã học" | Các mô hình nhỏ huấn luyện trên sở thích con người; dùng làm giám khảo tự động. |
| Elo | "Hệ số Elo cờ vua" | Tổng hợp Bradley-Terry từ các trận thắng theo cặp. |
| PartiPrompts | "Bộ prompt benchmark" | 1.600 prompt do Google tuyển chọn qua 12 hạng mục. |
| FD-DINO | "Vật thay thế tự giám sát" | FD dùng đặc trưng DINOv2; tốt hơn cho các miền ngoài ImageNet. |

## Ghi chú production: đánh giá cũng là một workload suy luận

Chạy FID trên 10k mẫu nghĩa là sinh 10k ảnh. Với SDXL base 50 bước ở 1024² trên một GPU L4, đó là khoảng ~11 giờ suy luận từng yêu cầu một. Ngân sách đánh giá là thật, và cách đặt vấn đề đúng y hệt kịch bản suy luận offline (tối đa hoá thông lượng, bỏ qua TTFT):

- **Batch mạnh, quên độ trễ.** Eval offline = static batching ở kích thước lớn nhất còn vừa bộ nhớ. `pipe(...).images` với `num_images_per_prompt=8` trên H100 80GB chạy nhanh hơn 4-6× về thời gian thực so với từng yêu cầu đơn lẻ.
- **Cache đặc trưng thật.** Việc trích đặc trưng Inception (FID) hoặc CLIP (CLIP-score, CMMD) trên tập tham chiếu thật chỉ chạy *một lần*, lưu thành `.npz`. Đừng tính lại mỗi lần eval.

Với cổng CI / hồi quy: chạy FID + CLIP score trên tập con 500 mẫu mỗi PR (~30 phút); chạy FID đầy đủ 10k + HPSv2 + Elo hằng đêm.

## Đọc thêm

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — bài báo FID.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — khảo sát các dạng lỗi.

---

[← Bài trước: Flow Matching & Rectified Flows](../../13-flow-matching-rectified-flows/docs/vi.md) · [Bài sau: Mô hình tự hồi quy thị giác (VAR): dự đoán thang kế tiếp →](../../19-visual-autoregressive-var/docs/vi.md)
