# CT-MAE baseline v1

## Protocol đã chốt

- OrganAMNIST 128×128, grayscale; train 34.561, validation 6.491, test 17.778.
- SSL bỏ nhãn trên toàn bộ train. Phân loại dùng cùng 1.728 index train được lấy
  phân tầng theo lớp, seed 42. Validation/test giữ đủ nhãn; 5% chỉ tính nhãn train.
- Input float32 / 255. Train dùng affine ±5°, dịch tối đa 5%, bilinear, fill 0,
  không flip/crop/scale/shear. Ảnh sau affine là target đầy đủ của MAE.
- Patch 8×8; 256 patch, che ngẫu nhiên đúng 192, encoder nhận 64 image token + CLS.
- Encoder 12 blocks, dimension 384, 6 heads; decoder 2 blocks, dimension 128, 4 heads.
  MLP ratio 4, fixed 2D sin/cos, LayerNorm epsilon 1e-6, MAE-style initialization.
- Masked-pixel MSE, `norm_pix_loss=false`, decoder tuyến tính. Reconstruction
  composite được ghi rõ là giữ vùng nhìn thấy và dự đoán vùng thiếu.
- Downstream dùng đủ patch, mean pooling bỏ CLS, Linear(384,11), Cross-Entropy.

| Stage | Epochs | LR cực đại | Physical batch × accumulation | Warmup | AdamW betas | WD |
|---|---:|---:|---|---:|---|---:|
| pretrain | 200 | 1.5e-4 | 64×4=256 | 10 | 0.9, 0.95 | 0.05 |
| scratch | 100 | 3e-4 | 32×2=64 | 5 | 0.9, 0.999 | 0.05 |
| linear | 100 | 1e-3 | 32×2=64 | 5 | 0.9, 0.999 | 0 |
| finetune | 100 | 3e-4 | 32×2=64 | 5 | 0.9, 0.999 | 0.05 |

AdamW epsilon 1e-8, không decay bias/LayerNorm. LR warmup rồi cosine xuống 1e-6.
AMP FP16 trên một GPU. Dropout 0; stochastic depth 0 cho MAE, tuyến tính 0→0.1
cho classifier. Linear probing đóng băng encoder và luôn giữ encoder eval.
Gradient accumulation tính theo số ảnh thực tế, kể cả microbatch/nhóm cuối thiếu.
Probe VRAM có hoàn tác weights/RNG; nếu OOM, batch giảm một nửa và accumulation
tăng gấp đôi. Recipe thực dùng được lưu riêng trong metadata và W&B.

## Chọn checkpoint và báo cáo

- Pretrain chọn min validation masked MSE; mask validation cố định theo index.
- Classifier chọn max validation macro-F1; bằng điểm giữ checkpoint trước.
- Không early stopping; test sau khi train đủ budget, dùng checkpoint được chọn.
- Accuracy, macro-F1, F1 từng lớp, confusion matrix và prediction/probability test
  được lưu từ inference thực. Đồ thị theo epoch dùng train/validation.
- Báo cáo `training_report.html` mở offline, `training_report.md`, các PNG và JSON
  giải thích cấu hình, điều chỉnh so với MAE, số tham số, tài nguyên và kết quả.
- Một seed là kết quả sơ bộ; không giả định SSL chắc chắn thắng supervised.

## Kaggle / W&B

Notebook: https://www.kaggle.com/code/lawlietwilllucky/ct-mae-baseline.
Chạy qua Kaggle MCP với private, GPU, Internet. Source/tests được zip và kiểm
SHA-256 ngay trong script; không phụ thuộc vào đính kèm input của notebook.
Raw data tải từ Zenodo, kiểm MD5 chính thức, lưu ở thư mục temp của Kaggle.

W&B project `ct-mae`; secret label `WANDB_API_KEY_THANG`. Mỗi stage là một run;
smoke có group/output/config riêng, dùng cùng kiến trúc nhưng chỉ 2 epochs,
tối đa 512 ảnh train và 128 validation, không đánh giá test chính thức.
Script chạy unit tests và kiểm tra CUDA RNG trước khi train; sau đó xác minh
summary của các run trên W&B API. `wandb_verification.json` lưu bằng chứng này.

Mỗi epoch ghi `last.pt` và metadata. Best weights lưu riêng. Mỗi 50 epochs và
cuối stage đồng bộ artifact gồm last/best/metadata để resume qua phiên Kaggle.
Checkpoint chứa model, optimizer, AMP scaler, RNG, epoch, lịch LR và history.
Thêm `--resume` khi tạo script Kaggle để tải artifact đúng protocol rồi tiếp tục.

## Interface và thao tác lại

Từ worktree, dùng `uv run --extra train python -m ct_mae.run --help`.
Các stage: `prepare`, `scratch`, `pretrain`, `linear`, `finetune`, `evaluate`,
`report`, `all`. `all` chạy scratch → pretrain → linear → finetune → test/report.
`--config` chọn recipe JSON, `--output` chọn experiment riêng. `--resume` dùng
checkpoint của stage có sẵn; `--stop-after-epoch N` chỉ áp dụng cho một stage.

Chạy CPU diagnostics bằng `--device cpu --wandb-mode disabled --smoke`; các lượt
production Kaggle giữ W&B online. `--stage report` chỉ đọc các artifact hiện có,
không tải dataset hay train lại. Recipe baseline được đóng gói cùng package;
Windows lock kiểm thử CPU và container Kaggle ghi lại các phiên bản runtime thực tế.

Nguồn: [MAE paper](https://arxiv.org/abs/2111.06377),
[official code](https://github.com/facebookresearch/mae/blob/main/models_mae.py),
[official recipe](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md),
[MedMNIST data](https://zenodo.org/records/10519652).
