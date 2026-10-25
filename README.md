# CT MAE

Học biểu diễn trên crop CT 2D bằng Transformer encoder và masked autoencoding.
Người dùng chủ động chốt thiết kế; agent implement và kiểm thử từng phần đã thống nhất.

Đã implement pipeline dữ liệu, asymmetric MAE, ba nhánh phân loại, checkpoint/resume,
W&B và báo cáo HTML/Markdown. Trạng thái và kết quả train thực tế nằm trong báo cáo;
smoke test được lưu riêng, không dùng thay cho kết quả baseline đầy đủ.

## Thiết kế đầu vào đã thống nhất

- Dataset: OrganAMNIST 2D, bản chính thức 128×128, một kênh xám.
- Patch size khởi đầu: 8×8, tạo lưới 16×16 = 256 patch mỗi ảnh.
- Người dùng chọn 128 vì bản 64 có cảm giác quá nhỏ sau khi xem ảnh;
  patch size có thể thay đổi trong các thí nghiệm sau.
- Tensor float32 `[B,1,128,128]`, pixel / 255; affine nhẹ ±5°, dịch tối đa 5%.
- MAE che 75% patch, encoder 12 blocks × 384 chiều, decoder 2 blocks × 128 chiều.
- SSL dùng toàn bộ ảnh train; cả ba nhánh phân loại dùng chung 1.728 ảnh có nhãn.
- Seed 42; pretrain 200 epochs, mỗi nhánh phân loại 100 epochs; chọn checkpoint bằng validation.

Protocol, cách chạy và giới hạn đánh giá được ghi trong [BASELINE.md](BASELINE.md).
Recipe máy đọc được nằm ở `src/ct_mae/baseline.json`. Báo cáo khám phá vẫn giữ pixel gốc.

## Chạy pipeline baseline

```powershell
Set-Location 'D:\Development\SchoolProject\ct-mae\.worktrees\dataset'
uv sync --extra train --group dev
uv run --extra train python -m ct_mae.run --stage prepare --data-dir 'D:\Development\SchoolProject\ct-mae\data' --output reports/baseline
uv run --extra train python -m ct_mae.kaggle --smoke --output work/kaggle_smoke.py
uv run --extra train python -m ct_mae.kaggle --output work/kaggle_baseline.py
```

Hai script cuối là bản đóng gói source/tests để submit qua **Kaggle MCP** vào notebook
[CT MAE Baseline](https://www.kaggle.com/code/lawlietwilllucky/ct-mae-baseline), private,
GPU và Internet bật. Dữ liệu được tải trực tiếp từ nguồn chính thức và kiểm MD5.
W&B dùng project `ct-mae`, đọc secret label `WANDB_API_KEY_THANG` đã cấp quyền trong Kaggle.
Chỉ label được lưu trong config; giá trị key không nằm trong repo hay script đóng gói.
Script production mới tự chạy và xác minh smoke trước khi bắt đầu full recipe;
không cần submit hai job GPU riêng. Bản `--smoke` chỉ chạy chẩn đoán.

Windows dùng PyTorch CPU trong venv để kiểm thử. Notebook giữ CUDA runtime của Kaggle,
ghi phiên bản thực tế vào `environment.json` và metadata từng run.

```powershell
uv run --extra train python -m ct_mae.run --stage report --output reports/baseline
uv run --extra train pytest -q
uv run --extra train ruff check .
uv run --extra train python scripts/check_training_report.py reports/baseline/training_report.html
```

## Repository và worktree

- Repository: `D:\Development\SchoolProject\ct-mae`
- Worktree: `D:\Development\SchoolProject\ct-mae\.worktrees\dataset`
- Branch triển khai: `feature/mae-baseline`
- Raw data dùng chung: `D:\Development\SchoolProject\ct-mae\data` (không đưa vào Git).

`SPEC.md` mô tả phạm vi và điều kiện nghiệm thu của milestone này.

## Chạy lại báo cáo trên Windows / PowerShell

Python 3.12 và `uv`. Các package được khóa trong `uv.lock`; phần khám phá dữ liệu chỉ
cần NumPy, Pillow và Matplotlib. Đọc trực tiếp file NPZ chính thức, không cần cài
PyTorch hay huấn luyện model để xem ảnh.

```powershell
Set-Location 'D:\Development\SchoolProject\ct-mae\.worktrees\dataset'
uv sync --locked --python 3.12 --group dev
uv run python -m ct_mae.download --data-dir 'D:\Development\SchoolProject\ct-mae\data' --sizes 28 64 128
uv run python -m ct_mae.explore --data-dir 'D:\Development\SchoolProject\ct-mae\data' --sizes 28 64 128 --output reports
Invoke-Item -LiteralPath '.\reports\explorer.html'
```

Download dùng URL, dung lượng và MD5 chính thức. File cache hợp lệ được dùng lại;
file không khớp checksum sẽ báo lỗi. Bản 224 được hỗ trợ bằng `--sizes 224`, nhưng
chưa tải trong lần khám phá này. Ba bản 28/64/128 tổng cộng khoảng 947 MB file nén.

Để đổi mẫu minh họa: thêm `--seed 7 --samples-per-class 8` khi sinh báo cáo.
Mặc định đóng gói 4 mẫu mỗi lớp ở mỗi split; thống kê vẫn tính trên toàn bộ dữ liệu.

## Những gì có trong báo cáo

- `explorer.html`: mở offline, chọn split/cơ quan/mẫu; so cùng index ở các resolution.
- Hai chế độ hiển thị: cùng khung 224 CSS px hoặc một pixel dữ liệu / một CSS pixel.
- `organ_samples.png`: ví dụ của cả 11 lớp có nhãn.
- `resolution_comparison.png`: cùng mẫu tại các resolution, hiển thị bằng khung bằng nhau.
- `dataset_overview.png`: phân bố lớp và histogram pixel của train.
- `dataset_summary.json`, `split_summary.csv`, `class_counts.csv`: số đo máy đọc được.

Ảnh minh họa giữ nguyên pixel uint8 và PNG lossless, dùng grayscale cố định 0..255.
Phóng to chỉ xảy ra khi hiển thị; không fit hay áp dụng preprocessing cho model.

## Các đặc tính đã đo từ file

Mỗi bản có 34.561 ảnh train, 6.491 validation và 17.778 test. Mọi ảnh có một nhãn
supervised thuộc 11 lớp. SSL sẽ bỏ qua nhãn trong pretraining; downstream dùng lại
nhãn của tập con được chọn sau này.

Tại mỗi resolution, toàn bộ ảnh có cùng H×W, một kênh, dtype uint8 và khoảng pixel
0..255. 128×128 có 4 lần số pixel và dung lượng array uint8 so với 64×64. Nhãn theo
index khớp hoàn toàn giữa ba bản đã tải. Quy trình nhà phát hành xác định chúng là
các resize của cùng crop nguồn; đối chiếu nhãn không phải kiểm chứng độc lập về
nguồn scan của từng ảnh.

Ảnh đã được abdominal windowing, crop và resize từ CT nguồn. Metadata theo từng
ảnh không có scan ID, bounding box gốc, HU gốc hay spacing, nên không suy ra kích
thước cơ quan theo mm hoặc coi hàng chục nghìn crop là số bệnh nhân độc lập.
Giữ nguyên split chính thức. Nếu sau này chọn mean/std cho model, chỉ fit trên train.

## Kiểm tra

```powershell
uv run --extra train pytest -q
uv run --extra train ruff check .
uv run python scripts/check_report.py reports/explorer.html --screenshot reports/browser_preview.png
```

Browser check dùng Microsoft Edge đã cài, chạy headless bằng Playwright. Nó kiểm tra
chọn class/split/mẫu, kích thước ảnh gốc, hiển thị 1:1, bàn phím và overflow tại các
viewport 320/768/1024/1440 px. Data, venv, work files và reports được loại khỏi Git.

## Nguồn và ghi công

- [MedMNIST](https://medmnist.com/)
- [Official data record](https://zenodo.org/records/10519652)
- [Metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
- [Cách tạo MedMNIST+](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md)

Dataset: CC BY 4.0. Khi công bố kết quả, trích dẫn MedMNIST v2 (Yang et al.,
Scientific Data, 2023) và các công trình dữ liệu nguồn theo hướng dẫn MedMNIST.

## Phân tích sau baseline

[FOLLOWUP.md](FOLLOWUP.md) mô tả thí nghiệm mở 3 block cuối từ cùng MAE checkpoint,
phân tích lỗi từng cơ quan/ảnh và PCA/SVD của encoder trên test. Báo cáo thực tế
nằm ở `D:\Development\SchoolProject\ct-mae\reports\followup\followup_report.html`;
mở offline, có bộ lọc worst cases, biểu đồ tương tác và 19 hình PNG xuất riêng.
Kết quả và giới hạn một seed được ghi trong tài liệu; dữ liệu, checkpoint và
report không đưa vào Git.

## So sánh 5% và 10% nhãn

[LABEL_STUDY.md](LABEL_STUDY.md) ghi protocol và kết quả của bốn nhánh ở mốc
3.456 ảnh có nhãn, chứa toàn bộ tập 1.728 ảnh cũ. Mỗi nhánh chạy 100 epoch,
seed 42, tái sử dụng chính MAE checkpoint đã có; không train lại MAE.

Ở 10%, supervised từ đầu đạt 84,98% Accuracy, SSL linear đạt 86,18%,
full fine-tuning đạt 91,66% và partial đạt 91,00%. Mốc 5% partial nhỉnh hơn full;
ở 10% thứ hạng đảo lại. Đây là kết quả một seed, cùng số epoch nhưng khác số
optimizer updates. Báo cáo có bảng từng organ, confusion matrix, đường học,
ảnh được sửa/ảnh mới sai và các giới hạn diễn giải.

Báo cáo offline:
`D:\Development\SchoolProject\ct-mae\reports\label-study\label_efficiency_report.html`.
Bản Markdown, JSON và các hình PNG nằm cùng thư mục.

## Đối chiếu crop cùng organ

[CROP_REVIEW.md](CROP_REVIEW.md) ghi cách kiểm tra pixel gốc và chọn ảnh cùng nhãn,
có bố cục tương tự nhưng kết quả dự đoán khác nhau. Trang offline
`reports/followup/crop_review.html` có 8 ảnh tham chiếu thận phải, ảnh đúng/được
sửa đúng để đối chiếu, chế độ pixel 1:1 và ba resolution 28/64/128. Dùng lại các
prediction đã có; đây là phân tích hình ảnh, chưa xác định nguyên nhân lỗi.

Đối chiếu riêng ảnh test #10267 với lớp tụy:

```powershell
uv run python scripts/visualize_pancreas.py --root .
uv run python scripts/check_pancreas_review.py reports/followup/pancreas_review.html
```

Trang `reports/followup/pancreas_review.html` có 15 mẫu ngẫu nhiên seed 42,
15 mẫu gần tham chiếu theo MSE bố cục pixel và toàn bộ 1.622 ảnh test mang nhãn
tụy. Hiển thị nhãn dataset cùng dự đoán của bốn model 5%; PNG giữ nguyên pixel.
Hai lưới ảnh lớn nằm trong `reports/followup/figures/pancreas_*_10267.png`.

## Học tiếp classifier: sampler, trọng số và SupCon

[ADAPTATION.md](ADAPTATION.md) ghi protocol và kết quả của mười lượt học tiếp:
năm biến thể cho full và năm cho partial, 50 epoch mỗi lượt, seed 42, cùng 1.728
nhãn gốc. So sánh CE học tiếp, cách ghép batch, trọng số CE theo organ và SupCon;
giữ checkpoint nguồn nếu validation không được cải thiện.

Validation chọn full/CE ở epoch 13: Accuracy 88,75%→89,09%, macro-F1 88,04%→88,68%.
Partial giữ nguồn 89,17% Accuracy; trong lượt này trọng số/SupCon chưa tạo ra
phương án tốt hơn theo validation. Báo cáo đối chiếu cả ảnh được sửa và ảnh mới
sai, F1 từng organ, confusion matrix, calibration và đặc trưng encoder.

Báo cáo offline: `reports/adaptation-study/adaptation_report.html`.
Artifacts đã tải và kiểm tra độc lập; W&B ghi đủ mười run. Kết quả một seed
được trình bày cùng giới hạn, không chọn lại mô hình theo test.
