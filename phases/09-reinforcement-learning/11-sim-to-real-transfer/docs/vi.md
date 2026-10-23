# Chuyển giao (transfer) sim-to-real (từ mô phỏng ra thực tế)

> Một policy huấn luyện trong mô phỏng mà thất bại trên phần cứng là policy đã học thuộc lòng mô phỏng. Ngẫu nhiên hoá miền, thích ứng miền và nhận dạng hệ thống là ba công cụ để bộ điều khiển học được vượt qua khoảng cách thực tế.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 9, Bài 08 (PPO), Giai đoạn 2, Bài 10 (Thiên lệch/phương sai)
**Time:** ~45 phút

## Vấn đề

Huấn luyện một robot thật thì chậm, nguy hiểm và tốn kém. Một robot hai chân cần hàng triệu episode huấn luyện mới học được cách đi; robot hai chân thật chỉ ngã một lần là hỏng phần cứng. Mô phỏng cho bạn số lần reset không giới hạn, khả năng tái lập tất định, môi trường song song và không hư hại gì.

Nhưng mô phỏng thì sai. Ổ trục có ma sát nhiều hơn mô hình MuJoCo. Camera có méo ống kính mà mô phỏng không tính tới. Các actuator (cơ cấu chấp hành) có độ trễ, khe hở và bão hoà mà 99% mô hình mô phỏng bỏ qua. Gió, bụi và ánh sáng thay đổi phá hỏng một policy huấn luyện trên hình ảnh sạch. **reality gap** (khoảng cách thực tế) — khác biệt có hệ thống giữa phân phối trong mô phỏng và phân phối thật — là vấn đề trung tâm của RL triển khai cho robot.

Bạn cần một policy *bền vững trước dịch chuyển phân phối sim-to-real*. Ba cách tiếp cận truyền thống: domain randomization (ngẫu nhiên hoá miền) — ngẫu nhiên hoá mô phỏng; domain adaptation (thích ứng miền) bằng một ít dữ liệu thật; hoặc system identification (nhận dạng hệ thống) — nhận dạng tham số của hệ thật rồi khớp mô phỏng theo. Năm 2026, công thức thống trị kết hợp cả ba với mô phỏng song song quy mô lớn (Isaac Sim, Isaac Lab, Mujoco MJX trên GPU).

## Khái niệm

![Ba chế độ sim-to-real: ngẫu nhiên hoá miền, thích ứng, nhận dạng hệ thống](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin và cộng sự 2017, Peng và cộng sự 2018. Trong lúc huấn luyện, hãy ngẫu nhiên hoá mọi tham số mô phỏng có thể khác trên robot thật: khối lượng, hệ số ma sát, gain PD của động cơ, sensor noise (nhiễu cảm biến), vị trí camera, ánh sáng, texture, mô hình tiếp xúc. Policy học một phân phối có điều kiện trên câu hỏi "hôm nay mình đang ở bản mô phỏng nào" và khái quát hoá trên toàn miền. Nếu robot thật nằm trong vùng bao huấn luyện thì policy chạy được.

- **Ưu điểm:** không cần dữ liệu thật. Một công thức, nhiều robot.
- **Nhược điểm:** huấn luyện ngẫu nhiên hoá quá mức tạo ra policy "vạn năng" nhưng quá thận trọng. Nhiễu quá nhiều ≈ chính quy hoá quá nhiều.

**System Identification (SI).** Khớp tham số của mô phỏng với dữ liệu thật trước khi huấn luyện. Nếu đo được ma sát khớp tay trên robot thật, hãy đưa giá trị đó vào mô phỏng. Rồi huấn luyện policy kỳ vọng đúng những giá trị ấy. Cần truy cập hệ thống thật nhưng giảm khoảng cách thực tế một cách trực tiếp.

- **Ưu điểm:** mục tiêu huấn luyện chính xác, ít nhiễu.
- **Nhược điểm:** sai số mô hình còn lại thì policy không thấy được; những hiệu ứng nhỏ chưa nhận dạng (ví dụ vùng chết của động cơ) vẫn làm hỏng triển khai.

**Domain Adaptation.** Huấn luyện trong mô phỏng, fine-tune với một ít dữ liệu thật. Có hai kiểu:

- **Real2Sim2Real:** học một mô phỏng phần dư `f(s, a, z) - f_sim(s, a)` từ các rollout thật, rồi huấn luyện trong mô phỏng đã hiệu chỉnh. Thu hẹp khoảng cách mà không cần nhiều dữ liệu thật.
- **Observation adaptation:** huấn luyện policy ánh xạ obs thật → obs giống trong mô phỏng nhờ bộ trích đặc trưng đã học (ví dụ GAN pixel-to-pixel). Bộ điều khiển vẫn nằm trong mô phỏng.

**Privileged learning / teacher-student.** Miki và cộng sự 2022 (robot bốn chân ANYmal). Huấn luyện một *teacher* trong mô phỏng, có quyền truy cập thông tin đặc quyền (ma sát, độ cao địa hình, độ trôi IMU đúng thực tế). Distill một *student* chỉ nhìn thấy quan sát từ cảm biến thật. Student học cách suy ra các đặc trưng đặc quyền từ lịch sử, bền vững qua các bộ tham số vật lý.

**Mô phỏng song song quy mô lớn.** 2024–2026. Isaac Lab, Mujoco MJX, Brax đều chạy hàng nghìn robot song song trên một GPU. PPO với 4.096 humanoid song song thu được nhiều năm kinh nghiệm trong vài giờ. "Khoảng cách thực tế" co lại khi phân phối huấn luyện mở rộng; DR gần như miễn phí khi mỗi env trong 4.096 env đó có tham số ngẫu nhiên khác nhau.

**Công thức thực tế năm 2026 (ví dụ robot bốn chân đi bộ):**

1. Mô phỏng song song quy mô lớn với trọng lực, ma sát, gain động cơ, tải trọng được ngẫu nhiên hoá.
2. Policy teacher huấn luyện với thông tin đặc quyền (bản đồ địa hình, vận tốc thân đúng thực tế).
3. Policy student distill từ teacher, chỉ dùng proprioception (encoder khớp chân).
4. Tuỳ chọn: thích ứng quan sát bằng autoencoder trên IMU thật.
5. Triển khai. Zero-shot trên hơn 10 môi trường. Nếu thất bại, fine-tune vài phút trên thực tế với PPO có ràng buộc an toàn.

```figure
f3-reality-gap
```

## Tự tay làm

Code của bài này là một minh hoạ nhỏ về ngẫu nhiên hoá miền trên một GridWorld có chuyển trạng thái *có nhiễu*. Ta huấn luyện một policy trải qua các xác suất trượt ngẫu nhiên trong "sim" rồi đánh giá trên "real" với mức trượt mà nó chưa từng thấy khi huấn luyện. Hình dạng bài toán ánh xạ trực tiếp sang chuyển giao từ MuJoCo sang phần cứng.

### Bước 1: mô phỏng có tham số

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` là tham số mà mô phỏng phơi ra. Trong robot thật nó có thể là ma sát, khối lượng, gain động cơ — bất cứ thứ gì thay đổi giữa mô phỏng và thực tế.

### Bước 2: huấn luyện với DR

Đầu mỗi episode, lấy mẫu `slip ~ Uniform[0.0, 0.4]`. Huấn luyện bằng PPO / Q-learning / gì cũng được. Làm vậy trong nhiều episode.

### Bước 3: đánh giá zero-shot trên các mức trượt "thật"

Đánh giá trên `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Bốn giá trị đầu nằm trong vùng hỗ trợ huấn luyện; `0.5` và `0.7` thì ở ngoài. Policy huấn luyện bằng DR nên gần tối ưu trong vùng hỗ trợ và giảm dần một cách mềm mại ở ngoài. Policy huấn luyện với slip cố định sẽ rất dễ vỡ khi ra ngoài mức trượt đã huấn luyện.

### Bước 4: so sánh với huấn luyện hẹp

Huấn luyện policy thứ hai chỉ với `slip = 0.0`. Đánh giá trên cùng dải `slip`. Bạn sẽ thấy kết quả sụt thảm hại ngay khi slip thật > 0.

## Cạm bẫy

- **Ngẫu nhiên hoá quá nhiều.** Huấn luyện trên `slip ∈ [0, 0.9]` thì policy của bạn né rủi ro đến mức không bao giờ thử đường tối ưu. Hãy khớp với phân phối thực tế *kỳ vọng*, không phải "cái gì cũng có thể xảy ra".
- **Ngẫu nhiên hoá quá ít.** Huấn luyện trên một lát mỏng thì policy không khái quát hoá được gì. Dùng curriculum thích ứng (Automatic Domain Randomization) mở rộng phân phối khi policy tiến bộ.
- **Nhận dạng sai không gian tham số.** Ngẫu nhiên hoá sai thứ (màu sắc camera trong khi khoảng cách thực tế là độ trễ động cơ) thì DR không giúp gì. Hãy đo đạc robot thật trước.
- **Rò rỉ thông tin đặc quyền.** Teacher dùng trạng thái toàn cục để chọn hành động, không chỉ quan sát, có thể tạo ra student không theo kịp. Hãy đảm bảo student thực hiện được policy của teacher chỉ với lịch sử quan sát.
- **Chuyển giao sim-to-sim thất bại.** Nếu policy không bền vững với một biến thể mô phỏng khó hơn thì nó cũng sẽ không bền vững với thế giới thật. Luôn kiểm thử trên một biến thể mô phỏng giữ riêng trước khi triển khai.
- **Không có vùng bao an toàn ngoài thực tế.** Policy chạy được trong mô phỏng và "chạy được ngoài thật" mà không có lớp khiên an toàn cấp thấp vẫn có thể làm hỏng phần cứng. Hãy thêm giới hạn tốc độ, giới hạn mô-men, giới hạn khớp trong một bộ điều khiển không học.

## Áp dụng

Bộ công cụ sim-to-real năm 2026:

| Miền | Bộ công cụ |
|--------|-------|
| Robot đi bằng chân (ANYmal, Spot, humanoid) | Isaac Lab + DR + teacher / student với thông tin đặc quyền |
| Thao tác (bàn tay khéo léo, gắp và đặt) | Isaac Lab + DR + DR-GAN cho thị giác |
| Lái xe tự động | CARLA / NVIDIA DRIVE Sim + DR + fine-tune trên thực tế |
| Đua drone | RotorS / Flightmare + DR + thích ứng trực tuyến |
| Thao tác bằng ngón tay / trong lòng bàn tay | OpenAI Dactyl (DR ở quy mô chưa từng có) |
| Tay máy công nghiệp | MuJoCo-Warp + SI + fine-tune nhỏ trên thực tế |

Với điều khiển ở mọi quy mô, quy trình đều nhất quán: khớp mô phỏng tốt nhất có thể, ngẫu nhiên hoá những gì không khớp được, huấn luyện policy thật lớn, distill, rồi triển khai kèm khiên an toàn.

## Đóng gói

Lưu thành `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Bài tập

1. **Dễ.** Huấn luyện một agent Q-learning trên GridWorld có slip cố định (slip=0.0). Đánh giá trên slip ∈ {0.0, 0.1, 0.3, 0.5}. Vẽ đồ thị return theo slip.
2. **Trung bình.** Huấn luyện một agent Q-learning dùng DR, lấy mẫu `slip ~ Uniform[0, 0.3]`. Đánh giá trên cùng dải. DR giúp được bao nhiêu tại slip=0.5 (ngoài phân phối)?
3. **Khó.** Cài đặt một curriculum: bắt đầu với slip=0.0, mở rộng dải DR mỗi khi policy đạt 90% mức tối ưu. Đo tổng số bước môi trường để đạt zero-shot tại slip=0.3 so với baseline DR cố định.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Reality gap | "Khác biệt sim-to-real" | Dịch chuyển phân phối về vật lý và cảm biến giữa lúc huấn luyện và lúc triển khai. |
| Domain randomization (DR) | "Huấn luyện qua nhiều mô phỏng ngẫu nhiên" | Ngẫu nhiên hoá tham số mô phỏng trong lúc huấn luyện để policy khái quát hoá. |
| System identification (SI) | "Đo hệ thật rồi khớp mô phỏng" | Ước lượng tham số vật lý thật; chỉnh mô phỏng cho khớp. |
| Domain adaptation | "Fine-tune trên dữ liệu thật" | Fine-tune một ít trên thực tế sau khi huấn luyện trong mô phỏng; có thể thích ứng quan sát hoặc động lực học. |
| Privileged info | "Ground truth dành cho teacher" | Thông tin chỉ mô phỏng mới có; student phải suy ra từ lịch sử quan sát. |
| Teacher/student | "Distill đặc quyền -> quan sát được" | Teacher huấn luyện với đường tắt; student học cách bắt chước mà không có chúng. |
| ADR | "Automatic Domain Randomization" | Curriculum mở rộng dải DR khi policy tiến bộ. |
| Real2Sim | "Thu hẹp khoảng cách bằng dữ liệu thật" | Học một phần dư để mô phỏng bắt chước các rollout thật. |

## Đọc thêm

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — bài báo DR gốc (thị giác cho robot).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR cho động lực học, robot bốn chân đi bộ.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR ở quy mô lớn.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — teacher-student cho ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — mô phỏng song song quy mô lớn đứng sau các đợt triển khai 2025–2026.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — phương pháp curriculum ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — cách đóng khung Dyna (dùng mô hình để lập kế hoạch + rollout) làm nền cho các pipeline sim-to-real hiện đại.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — phân loại các phương pháp sim-to-real kèm kết quả benchmark.

---

[← Bài trước: RL đa tác tử](../../10-multi-agent-rl/docs/vi.md) · [Bài sau: RL cho game →](../../12-rl-for-games/docs/vi.md)
