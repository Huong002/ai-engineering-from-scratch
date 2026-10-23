# AI Scientist v2 — Nghiên cứu tự chủ ở cấp workshop

> AI Scientist v2 của Sakana (Yamada et al., arXiv:2504.08066) thực hiện toàn bộ vòng nghiên cứu: giả thuyết, code, thí nghiệm, hình, bài viết, nộp bài. Đây là hệ thống đầu tiên có bài báo do AI tạo vượt qua bình duyệt tại một workshop của ICLR 2025. Đánh giá độc lập (Beel et al.) cho thấy 42% thí nghiệm thất bại do lỗi code, còn bước rà soát tài liệu thường nhận nhầm khái niệm đã có là mới. Tài liệu của chính Sakana cảnh báo codebase chạy code do LLM viết và khuyên cách ly bằng Docker. Cả hai mặt ấy đều quan trọng.

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Giai đoạn 15, Bài 03 (AlphaEvolve), Giai đoạn 15, Bài 04 (DGM)
**Time:** ~60 phút

## Vấn đề

Nghiên cứu là nhiệm vụ mở. Khác với tìm kiếm thuật toán của AlphaEvolve hay việc DGM tự sửa đổi trong giới hạn benchmark, kết quả nghiên cứu không có tiêu chí đúng sai để máy tự kiểm tra. Người bình duyệt đánh giá bài báo, không phải unit test. Vì thế, khó khép kín vòng lặp hơn — nhưng nếu làm được, giá trị cũng lớn hơn, bởi nghiên cứu là nơi tiến bộ tích lũy.

AI Scientist v1 (Sakana, 2024) khép kín vòng lặp nhờ các mẫu do con người viết. LLM điền thí nghiệm vào khung cố định. AI Scientist v2 (Yamada et al., 2025) bỏ yêu cầu dùng mẫu bằng agentic tree search (tìm kiếm cây do agent điều khiển) cùng vòng phản biện của mô hình thị giác-ngôn ngữ. Hệ thống đề xuất ý tưởng, triển khai thí nghiệm, tạo hình, viết bài và chỉnh sửa theo nhận xét bình duyệt.

Kết quả bình duyệt: một bài do v2 tạo được nhận tại workshop của ICLR 2025 (có công bố nguồn gốc). Kết quả đánh giá độc lập: hệ thống còn xa mới đáng tin cậy. Cả hai đều đúng.

## Khái niệm

### Kiến trúc

1. **Đề xuất ý tưởng.** LLM đề xuất hướng nghiên cứu dựa trên chủ đề và tài liệu trước đó. v1 dùng mẫu; v2 tìm kiếm trong không gian giả thuyết bằng agentic search (tìm kiếm do agent điều khiển).
2. **Kiểm tra tính mới.** Bước truy xuất tài liệu kiểm tra ý tưởng đã được công bố chưa. Đây là bước mà đánh giá của Beel et al. phát hiện lỗi nhận nhầm — các phương pháp đã có thường bị xếp là mới.
3. **Lập kế hoạch thí nghiệm.** Agent soạn quy trình thí nghiệm và viết code.
4. **Thực thi.** Code chạy trong sandbox (môi trường cách ly). Lỗi được đưa lại vào vòng thử lại. Theo đo lường của Beel et al., 42% thí nghiệm thất bại vì lỗi code ở bước này.
5. **Tạo hình.** Mô hình thị giác-ngôn ngữ đọc hình đã tạo và chỉnh sửa cho rõ hơn. Đây là bổ sung kỹ thuật chủ chốt của v2.
6. **Viết bài.** LLM soạn bài báo và chỉnh sửa qua phản biện nội bộ.
7. **Tùy chọn: nộp bài.** Bài báo được nộp tới một hội nghị hoặc workshop.

### Kết quả được nhận tại workshop có ý nghĩa gì

Một bài do v2 tạo đã vượt qua bình duyệt tại workshop của ICLR 2025. Các tác giả đã báo với ban chương trình về nguồn gốc bài báo. Việc được nhận là một điểm dữ liệu, không phải căn cứ để khẳng định hệ thống “làm nghiên cứu”.

Bối cảnh quan trọng: yêu cầu với bài workshop thấp hơn bài ở hội nghị chính. Bình duyệt có độ nhiễu; vào một ngày bất kỳ, chỉ một phần nhỏ bài nộp được nhận. Một thành công chứng minh tính khả thi, không chứng minh độ tin cậy. Bài báo trên Nature 2026 mô tả vòng lặp từ đầu đến cuối và có các nhà nghiên cứu là con người đồng tác giả; không thể nói “hệ thống đã viết một bài trên Nature”.

### Đánh giá độc lập phát hiện gì

Beel et al. (arXiv:2502.14297) thực hiện đánh giá bên ngoài. Các phát hiện chính:

- **Thí nghiệm thất bại.** 42% thí nghiệm thất bại vì lỗi code (import sai, kích thước không khớp, biến chưa định nghĩa). Vòng thử lại bắt được một số lỗi, nhưng không phải tất cả.
- **Nhận nhầm tính mới.** Bước truy xuất tài liệu thường đánh dấu khái niệm đã có là mới. Đây là dạng ảo giác trong nghiên cứu.
- **Khoảng cách về chất lượng trình bày.** Bước phản biện hình bằng mô hình thị giác-ngôn ngữ tạo hình đạt chuẩn xuất bản, che lấp điểm yếu của thí nghiệm.

Phát hiện cuối quan trọng với giai đoạn này. Hệ thống tạo đầu ra thuyết phục mà không làm nghiên cứu thuyết phục thì nguy hiểm hơn, chứ không an toàn hơn, hệ thống thất bại rõ ràng. Đánh giá phải kiểm tra các luận điểm nền tảng, không dừng ở hình.

### Nguy cơ vượt khỏi sandbox

README trong repo của chính Sakana cảnh báo:

> Do phần mềm này chạy code do LLM tạo, chúng tôi không thể bảo đảm an toàn. Có nguy cơ dùng gói nguy hiểm, truy cập web không kiểm soát và khởi tạo tiến trình ngoài ý muốn. Hãy tự cân nhắc rủi ro khi sử dụng và xem xét cách ly bằng Docker.

Đây là hình dạng vận hành của tự chủ trong một lĩnh vực chưa được kiểm chứng. LLM viết code; code chạy; code có thể làm mọi điều mà tiến trình được phép làm. Nếu sandbox không giới hạn cứng hệ thống tệp, mạng và các hành động của tiến trình, bất kỳ agent nghiên cứu tự định hướng nào cũng có thể lấy cắp dữ liệu, tiêu tốn tài nguyên tính toán hoặc tự viết lại.

Việc cách ly AlphaEvolve dễ hơn vì bộ đánh giá của nó có phạm vi chặt chẽ. Vòng lặp của AI Scientist v2 chạy code và theo đuổi mục tiêu đều không giới hạn trước. Vì vậy, nó cần cách ly mạnh hơn (tối thiểu Docker; nên dùng seccomp / gVisor) và rà soát thủ công từng bài trước khi gửi ra ngoài hệ thống.

### Vị trí của v2 trong nhóm hệ thống tiên phong

| Hệ thống | Mục tiêu | Dạng đầu ra | Bộ đánh giá | Lỗi đã biết |
|---|---|---|---|---|
| AlphaEvolve | thuật toán | code | unit + benchmark | phụ thuộc độ nghiêm ngặt của bộ đánh giá |
| DGM | khung agent | code | SWE-bench | khai thác sai lệch phần thưởng |
| AI Scientist v2 | bài báo nghiên cứu | văn bản + code + hình | bình duyệt (yếu) | thí nghiệm thất bại, nhận nhầm, trình bày đẹp che điểm yếu |

Trong ba hệ thống, v2 có bộ đánh giá tự động yếu nhất, phạm vi đầu ra rộng nhất và con đường ngắn nhất để tạo sản phẩm công khai. Các biện pháp vận hành (sandbox, rà soát, công bố nguồn gốc) đảm nhiệm phần lớn việc bảo đảm an toàn.

```figure
mx-research-loop
```

## Áp dụng

`code/main.py` mô phỏng vòng lặp v2 bằng máy trạng thái: ý tưởng → kiểm tra tính mới → thí nghiệm → hình → viết bài → bình duyệt → nhận hoặc lặp lại. Mỗi trạng thái có xác suất thất bại tùy chỉnh dựa trên phát hiện của Beel et al. Chạy mô phỏng N vòng và đếm:

- Có bao nhiêu ý tưởng đến bước nộp bài.
- Có bao nhiêu bài nộp mắc lỗi thí nghiệm nghiêm trọng bị bài viết trau chuốt che đi.
- Giới hạn số lần thử lại đánh đổi chất lượng với sản lượng ra sao.

## Đóng gói

`outputs/skill-ai-scientist-sandbox-review.md` là danh sách kiểm tra hai cửa cho mọi sản phẩm của agent nghiên cứu trước khi ra khỏi sandbox.

## Bài tập

1. Chạy `code/main.py` với tham số mặc định. Bao nhiêu phần trăm lượt chạy tạo bài báo “không lỗi”? Bao nhiêu phần trăm tạo bài có lỗi thí nghiệm bị bước chỉnh hình che đi?

2. Giá trị mặc định đã dùng mức 42% / 25% của Beel et al. Chạy lại với `--experiment-failure 0.20 --novelty-mislabel 0.10`, rồi với `--experiment-failure 0.60 --novelty-mislabel 0.40`. Tỷ lệ bài trau chuốt nhưng có lỗi thay đổi thế nào giữa hai lần chạy?

3. Đọc README của repo AI Scientist v2 từ Sakana về yêu cầu sandbox. Nêu hai hạn chế bổ sung (ngoài Docker) bạn sẽ áp dụng cho một lượt chạy tự chủ kéo dài nhiều ngày.

4. Đọc Mục 4 của Beel et al. về khoảng cách chất lượng trình bày. Thiết kế thêm một bộ đánh giá để phát hiện bài trông đẹp nhưng có lỗi thí nghiệm.

5. Đề xuất quy trình con người rà soát đầu ra của agent nghiên cứu, có khả năng mở rộng tốt hơn cách “một tiến sĩ đọc từng bài”. Xác định nút thắt và thiết kế cách xử lý.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| AI Scientist v1 | “Agent nghiên cứu dùng mẫu của Sakana” | Điền thí nghiệm vào khung cố định |
| AI Scientist v2 | “Agent nghiên cứu không cần mẫu” | Tìm kiếm cây do agent điều khiển, có VLM phản biện hình |
| Agentic tree search | “Agent nghiên cứu phân nhánh” | Mở rộng song song nhiều kế hoạch thí nghiệm; loại nhánh bằng bộ phản biện nội bộ |
| Vision-language critique | “VLM chỉnh hình cho đẹp” | Mô hình đa phương thức đọc và chỉnh hình cho rõ hơn |
| Literature retrieval | “Kiểm tra tính mới” | Tìm công trình trước đó để xác nhận ý tưởng mới — đã ghi nhận lỗi nhận nhầm |
| Polish masking | “Bài đẹp, nghiên cứu lỗi” | Chất lượng trình bày vượt chất lượng thí nghiệm, che giấu điểm yếu |
| Sandbox escape | “Code của LLM vượt khỏi môi trường cách ly” | Code do agent chạy thực hiện điều người thiết kế vòng lặp không định làm |

## Đọc thêm

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — bài báo.
- [Blog Sakana về bài công bố trên Nature 2026](https://sakana.ai/ai-scientist-nature/) — tóm tắt từ nhà cung cấp, có bối cảnh bình duyệt.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — số liệu đánh giá độc lập.
- [Bài báo về Sakana AI Scientist v1](https://arxiv.org/abs/2408.06292) — phiên bản tiền nhiệm dùng mẫu.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn rộng hơn về agent nghiên cứu với nhiệm vụ mở.

---

[← Bài trước: Darwin Godel Machine](../../04-darwin-godel-machine/docs/vi.md) · [Bài sau: Nghiên cứu căn chỉnh tự động (Anthropic AAR) →](../../06-automated-alignment-research/docs/vi.md)
