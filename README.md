# CT MAE

Học biểu diễn trên crop CT 2D bằng Transformer encoder và masked autoencoding.
Người dùng chủ động chốt thiết kế; agent implement và kiểm thử từng phần đã thống nhất.

Milestone hiện tại: xem dữ liệu OrganAMNIST nguyên trạng để quyết định preprocessing.
Resolution đầu vào, patch size, normalization, augmentation và ngân sách nhãn còn mở.
Các giá trị hyperparameter đã thảo luận là ứng viên, chưa phải cấu hình đã chốt.

## Repository và worktree

- Repository: `D:\Development\SchoolProject\ct-mae`
- Worktree: `D:\Development\SchoolProject\ct-mae\.worktrees\dataset`
- Branch: `feature/dataset-explorer`
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
uv run pytest -q
uv run ruff check .
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
