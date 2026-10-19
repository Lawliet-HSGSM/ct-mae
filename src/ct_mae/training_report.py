"""Generate an offline configuration-and-results report from recorded experiment files."""

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ct_mae.download import METADATA

NAMES = {
    "pretrain": "MAE pretraining",
    "scratch": "Supervised baseline",
    "linear": "SSL linear probing",
    "finetune": "SSL fine-tuning",
}


def configuration_rows(config: dict) -> list[dict]:
    model, augmentation, training = config["model"], config["augmentation"], config["training"]
    grid = model["image_size"] // model["patch_size"]
    visible = int(grid**2 * (1 - model["mask_ratio"]))
    betas = "; ".join(f"{name}: {recipe['betas']}" for name, recipe in config["stages"].items())
    decay = "; ".join(
        f"{name}: {recipe['weight_decay']:g}" for name, recipe in config["stages"].items()
    )
    values = [
        (
            "Dataset / resolution",
            f"OrganAMNIST {model['image_size']}×{model['image_size']}, 1 kênh",
            "Crop cơ quan CT 2D ở kích thước cố định.",
            "Giữ thêm chi tiết so với bản 64 pixel.",
            "Resolution ảnh hưởng thông tin đầu vào, số patch và chi phí tính toán.",
            "MedMNIST+ / project",
        ),
        (
            "Ngân sách nhãn",
            f"{config['dataset']['labeled_count']} ảnh train, khoảng 5%",
            "MAE dùng toàn bộ ảnh train; phân loại dùng chung một tập con có nhãn.",
            "Đo hiệu quả chuyển giao trong điều kiện ít nhãn.",
            "Full labeled validation được giữ riêng; chi phí nhãn validation không tính trong 5%.",
            "Thiết kế project",
        ),
        (
            "Seed",
            str(config["seed"]),
            "Điểm khởi đầu cho các bộ sinh số ngẫu nhiên.",
            "Tái lập tập con, khởi tạo, thứ tự dữ liệu và masking.",
            "Một seed cho kết quả sơ bộ; chưa đo độ dao động giữa nhiều lượt chạy.",
            "Thiết kế project",
        ),
        (
            "Tensor / intensity",
            "float32, [B,1,H,W], pixel / 255 → [0,1]",
            "Biểu diễn ảnh bằng số thực, giữ nguyên thang độ sáng tương đối.",
            "MSE có đơn vị rõ ràng và preview dễ diễn giải.",
            "Thang pixel quyết định độ lớn loss/gradient; không dùng mean/std ImageNet.",
            "Điều chỉnh CT",
        ),
        (
            "Augmentation",
            f"Xoay ±{augmentation['degrees']:g}°, dịch ±{augmentation['translate']:.0%}",
            "Random affine, bilinear, fill 0; scale 1, không shear, flip hoặc crop.",
            "Biến đổi nhẹ trên train, bảo toàn phần lớn crop và phân biệt trái/phải.",
            "Target là ảnh sau affine; masking diễn ra sau đó. Val/test không augmentation.",
            "Điều chỉnh CT",
        ),
        (
            "Patch size",
            f"{model['patch_size']}×{model['patch_size']}; {grid}×{grid}={grid**2} patch",
            "Mỗi patch trở thành một token sau phép chiếu tuyến tính.",
            "Điểm khởi đầu đã chọn sau khi xem dữ liệu.",
            "Patch nhỏ tăng độ phân giải theo không gian và độ dài chuỗi attention.",
            "Điều chỉnh MAE",
        ),
        (
            "Mask ratio",
            f"{model['mask_ratio']:.0%}; giữ {visible} image token + CLS",
            "Chọn ngẫu nhiên patch bị loại khỏi encoder, riêng cho từng ảnh/lần train.",
            "Tạo bài toán khôi phục theo ngữ cảnh với encoder xử lý ít token hơn.",
            "Mask nhiều tăng độ khó; val dùng mask cố định theo index để so loss công bằng.",
            "MAE",
        ),
        (
            "Encoder",
            f"{model['encoder_depth']} blocks · D={model['encoder_dim']} · "
            f"{model['encoder_heads']} heads",
            "ViT học biểu diễn dùng lại cho phân loại.",
            "Encoder là phần chính; chọn quy mô vừa tài nguyên cá nhân.",
            "Depth/width tăng năng lực, số tham số, VRAM và thời gian train.",
            "Điều chỉnh MAE",
        ),
        (
            "Decoder",
            f"{model['decoder_depth']} blocks · D={model['decoder_dim']} · "
            f"{model['decoder_heads']} heads",
            "Decoder nhỏ tái tạo đủ vị trí patch từ latent và mask token.",
            "Giữ thiết kế asymmetric; decoder được bỏ khi phân loại.",
            "Decoder vẫn xử lý đầy đủ chuỗi; độ nhỏ của nó giảm chi phí pretraining.",
            "Điều chỉnh MAE",
        ),
        (
            "MLP / normalization",
            f"MLP ratio {model['mlp_ratio']:g}; LayerNorm eps {model['layer_norm_eps']:g}",
            "Pre-norm Transformer với GELU và residual connections.",
            "Theo cấu trúc Transformer của MAE.",
            "MLP ratio điều khiển chiều rộng lớp ẩn; epsilon giúp ổn định phép chuẩn hóa.",
            "MAE",
        ),
        (
            "Khởi tạo / vị trí",
            "Xavier; CLS/mask normal std 0.02; fixed 2D sin/cos",
            "Encoder bắt đầu ngẫu nhiên, vị trí patch được cung cấp tường minh.",
            "Theo cách khởi tạo của mã MAE chính thức.",
            "Khởi tạo ảnh hưởng ổn định ban đầu; decoder không nhận feature trung gian từ encoder.",
            "MAE",
        ),
        (
            "Reconstruction loss",
            f"Masked MSE; norm_pix_loss={model['norm_pix_loss']}",
            "Trung bình sai số bình phương trên pixel của patch bị che.",
            "Baseline khôi phục trực tiếp intensity để kiểm tra định tính.",
            "Visible pixels không góp vào loss; decoder tuyến tính, chỉ clamp khi hiển thị.",
            "MAE pixel-target baseline",
        ),
        (
            "Classifier",
            f"Mean patch pooling → Linear({model['encoder_dim']},11)",
            "Dùng ảnh đầy đủ và loại CLS khỏi phép lấy trung bình.",
            "Cùng encoder/head cho cả ba nhánh; head khởi tạo std 0.02, bias 0.",
            "Linear probing chỉ cập nhật head; fine-tuning và scratch cập nhật toàn mạng.",
            "Thiết kế project",
        ),
        (
            "Optimizer",
            "AdamW; eps=" + str(training["adam_eps"]),
            "Ước lượng moment gradient và weight decay tách rời.",
            "Tham chiếu recipe MAE.",
            betas + ". Betas ảnh hưởng độ phản ứng của moment.",
            "MAE / điều chỉnh downstream",
        ),
        (
            "Weight decay",
            decay + "; bỏ bias và LayerNorm",
            "Regularization trên các trọng số được train.",
            "Theo cách chia nhóm tham số của MAE.",
            "Ảnh hưởng overfitting; linear probing chỉ có lớp head nhỏ.",
            "MAE / project",
        ),
        (
            "Learning rate / schedule",
            f"Warmup → cosine → {training['min_lr']:g}",
            "LR trong bảng stage là LR thực tế, không nhân ngầm theo batch.",
            "Warmup giảm bước nhảy đầu train; cosine giảm dần tốc độ cập nhật.",
            "Scratch và fine-tuning dùng cùng recipe; "
            "các LR/epochs được điều chỉnh cho baseline này.",
            "Điều chỉnh MAE",
        ),
        (
            "Batch / gradient accumulation",
            "Effective batch = physical batch × accumulation",
            "Gom gradient nhiều microbatch trước một lần optimizer update.",
            "Giữ batch hiệu dụng khi VRAM giới hạn.",
            "Loss được trọng số theo số ảnh, kể cả nhóm cuối thiếu ảnh; "
            "OOM giảm batch và tăng accumulation.",
            "PyTorch AMP / MAE",
        ),
        (
            "AMP / stochastic depth",
            f"AMP FP16={training['amp']}; drop path "
            f"0 (MAE), {model['classifier_drop_path']:g} (classifier)",
            "Mixed precision với GradScaler; dropout 0, "
            "stochastic depth tăng tuyến tính theo block.",
            "AMP tiết kiệm VRAM; stochastic depth regularize encoder được fine-tune/train từ đầu.",
            "Linear probe giữ encoder eval; "
            "log cả AMP scale và số optimizer step bị bỏ do overflow.",
            "PyTorch / điều chỉnh MAE",
        ),
        (
            "Chọn checkpoint",
            "Min val masked MSE / max val macro-F1",
            "Chọn theo validation, giữ checkpoint trước khi bằng điểm.",
            "Macro-F1 cho trọng số ngang nhau giữa 11 lớp có số ảnh khác nhau.",
            "Chạy đủ epochs; test chỉ sau khi hoàn tất và chọn checkpoint.",
            "Thiết kế project",
        ),
    ]
    keys = ("setting", "value", "meaning", "reason", "effect", "source")
    return [dict(zip(keys, row, strict=True)) for row in values]


def table(headers, rows) -> tuple[str, str]:
    web = (
        "<div class='scroll'><table><thead><tr>"
        + "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
        + "</tr></thead><tbody>"
    )
    web += (
        "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows
    )
    return web, "\n".join(lines)


def embedded_image(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def plot_histories(output: Path, runs: dict, results: dict) -> list[tuple[str, Path]]:
    figures = []
    active = {stage: run for stage, run in runs.items() if run.get("history")}
    if active:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for axis, stage in zip(axes.flat, NAMES, strict=True):
            history = active.get(stage, {}).get("history", [])
            for split in ("train", "val"):
                axis.plot(
                    [row["epoch"] for row in history],
                    [row[split]["loss"] for row in history],
                    label=split,
                )
            axis.set(
                title=NAMES[stage],
                xlabel="Epoch",
                ylabel="Masked MSE" if stage == "pretrain" else "Cross-entropy",
            )
            axis.grid(alpha=0.2)
            axis.legend()
        path = output / "training_curves.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        figures.append(("Loss theo epoch (train / validation)", path))
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for stage in ("scratch", "linear", "finetune"):
            history = active.get(stage, {}).get("history", [])
            for axis, metric in zip(axes, ("accuracy", "macro_f1"), strict=True):
                axis.plot(
                    [row["epoch"] for row in history],
                    [row["val"][metric] * 100 for row in history],
                    label=NAMES[stage],
                )
                axis.set(xlabel="Epoch", ylabel=f"Validation {metric} (%)", ylim=(0, 100))
                axis.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        path = output / "validation_comparison.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        figures.append(("So sánh trên validation trong quá trình train", path))
    if results:
        figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
        x = np.arange(len(results))
        for offset, metric in [(-0.18, "accuracy"), (0.18, "macro_f1")]:
            bars = axis.bar(
                x + offset,
                [result[metric] * 100 for result in results.values()],
                width=0.35,
                label=metric,
            )
            axis.bar_label(bars, fmt="%.1f", fontsize=9)
        axis.set(
            xticks=x,
            xticklabels=[NAMES[stage] for stage in results],
            ylim=(0, 110),
            ylabel="Test score (%)",
        )
        axis.legend()
        path = output / "test_comparison.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        figures.append(("Test của checkpoint được chọn bằng validation", path))
    for stage, result in results.items():
        if "confusion_matrix" not in result:
            continue
        confusion = np.array(result["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(8, 7), layout="constrained")
        image = axis.imshow(confusion, cmap="Blues")
        axis.set(
            xticks=range(11),
            yticks=range(11),
            xticklabels=METADATA["labels"],
            yticklabels=METADATA["labels"],
            xlabel="Predicted",
            ylabel="True",
            title=NAMES[stage],
        )
        plt.setp(axis.get_xticklabels(), rotation=60, ha="right", fontsize=8)
        for y in range(11):
            for x in range(11):
                axis.text(x, y, str(confusion[y, x]), ha="center", va="center", fontsize=6)
        figure.colorbar(image, ax=axis, shrink=0.7)
        path = output / f"confusion_{stage}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        figures.append((f"Confusion matrix: {NAMES[stage]}", path))
    return figures


def build_report(output: Path, config: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    runs, results = {}, {}
    for stage in NAMES:
        path = output / stage / "run.json"
        if path.exists():
            runs[stage] = json.loads(path.read_text("utf-8"))
            test_path = output / stage / "test.json"
            if (
                runs[stage]["status"] == "completed"
                and not runs[stage].get("smoke")
                and test_path.exists()
            ):
                results[stage] = json.loads(test_path.read_text("utf-8"))
    blocks, markdown = [], ["# CT-MAE — cấu hình và kết quả thực nghiệm"]

    def section(title: str, web: str, text: str):
        blocks.append(f"<section><h2>{html.escape(title)}</h2>{web}</section>")
        markdown.append(f"## {title}\n\n{text}")

    summary = (
        "OrganAMNIST 128×128 · patch 8×8 · asymmetric MAE · seed "
        + str(config["seed"])
        + ". Các giá trị chưa chạy là recipe dự kiến; kết quả chỉ lấy từ run đã ghi nhận."
    )
    section("Mục tiêu và trạng thái", f"<p>{html.escape(summary)}</p>", summary)
    protocol = (
        "SSL dùng 34.561 ảnh train và bỏ qua nhãn. Ba nhánh phân loại dùng chung "
        f"{config['dataset']['labeled_count']} ảnh có nhãn. Validation 6.491 và test 17.778 "
        "giữ nguyên. Ngân sách 5% chỉ áp dụng cho nhãn train. Một seed cho kết quả sơ bộ."
    )
    section("Dữ liệu và phép so sánh", f"<p>{html.escape(protocol)}</p>", protocol)
    headers = [
        "Giai đoạn",
        "Trạng thái",
        "Epoch đã chạy / budget",
        "LR cực đại",
        "Batch × tích lũy",
        "Warmup",
        "Weight decay",
    ]
    rows = []
    for stage, name in NAMES.items():
        run = runs.get(stage, {})
        recipe = run.get("recipe", config["stages"][stage])
        rows.append(
            [
                name,
                run.get("status", "Chưa chạy"),
                f"{run.get('completed_epochs', 0)} / {recipe['epochs']}",
                f"{recipe['lr']:g}",
                f"{recipe['batch_size']} × {recipe['accumulation']} = "
                f"{recipe['batch_size'] * recipe['accumulation']}",
                recipe["warmup_epochs"],
                recipe["weight_decay"],
            ]
        )
    web, text = table(headers, rows)
    section("Recipe và tiến độ thực tế", web, text)
    config_rows = configuration_rows(config)
    web, text = table(
        ["Cấu hình", "Giá trị", "Ý nghĩa", "Lý do", "Ảnh hưởng", "Nguồn"],
        [list(row.values()) for row in config_rows],
    )
    section("Giải thích các cấu hình ảnh hưởng training/kết quả", web, text)
    if results:
        web, text = table(
            ["Mô hình", "Accuracy", "Macro-F1", "Epoch được chọn"],
            [
                [
                    NAMES[stage],
                    f"{result['accuracy']:.2%}",
                    f"{result['macro_f1']:.2%}",
                    result.get("selected_epoch", "—"),
                ]
                for stage, result in results.items()
            ],
        )
        if "scratch" in results and "finetune" in results:
            difference = 100 * (results["finetune"]["accuracy"] - results["scratch"]["accuracy"])
            note = f"Fine-tuning − supervised: {difference:+.2f} điểm phần trăm Accuracy."
            web += f"<p>{note}</p>"
            text += "\n\n" + note
        section("Kết quả test thực tế", web, text)
        web, text = table(
            ["Cơ quan"] + [NAMES[stage] for stage in results],
            [
                [label] + [f"{result['per_class_f1'][index]:.2%}" for result in results.values()]
                for index, label in enumerate(METADATA["labels"])
            ],
        )
        section("F1 từng lớp trên test", web, text)
    else:
        section(
            "Kết quả test",
            "<p>Chưa có kết quả test của lượt train đầy đủ.</p>",
            "Chưa có kết quả test của lượt train đầy đủ.",
        )
    figures = []
    if (output / "input_preview.png").exists():
        figures.append(("Ảnh thật → affine target → input bị che", output / "input_preview.png"))
    if (output / "pretrain" / "reconstruction.png").exists():
        figures.append(
            (
                "Khôi phục: hàng cuối ghép pixel nhìn thấy với dự đoán vùng thiếu",
                output / "pretrain" / "reconstruction.png",
            )
        )
    figures.extend(plot_histories(output, runs, results))
    for caption, path in figures:
        section(
            caption,
            f"<img alt='{html.escape(caption)}' src='{embedded_image(path)}'>",
            f"![{caption}]({path.relative_to(output).as_posix()})",
        )
    evidence = []
    for stage, run in runs.items():
        evidence.append(
            {
                "stage": stage,
                "environment": run.get("environment"),
                "parameters": run.get("parameters"),
                "subset_sha256": run.get("subset_sha256"),
                "ssl_checkpoint_sha256": run.get("ssl_checkpoint_sha256"),
                "best_epoch": run.get("best_epoch"),
                "wandb_url": run.get("wandb_url"),
                "batch_adjustments": run.get("batch_adjustments", []),
            }
        )
    details = json.dumps(evidence, ensure_ascii=False, indent=2)
    section(
        "Dấu vết thực nghiệm và tài nguyên",
        f"<pre>{html.escape(details)}</pre>",
        f"```json\n{details}\n```",
    )
    references = [
        ("MAE paper", "https://arxiv.org/abs/2111.06377"),
        ("MAE implementation", "https://github.com/facebookresearch/mae/blob/main/models_mae.py"),
        ("MAE training recipe", "https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md"),
        ("Official OrganAMNIST data", METADATA["record"]),
    ]
    section(
        "Nguồn và các điều chỉnh",
        "<p>Kiến trúc, resolution, augmentation, epochs, batch và "
        "recipe downstream đã điều chỉnh cho OrganAMNIST; đây là baseline tham chiếu MAE.</p>"
        + "<ul>"
        + "".join(f"<li><a href='{url}'>{title}</a></li>" for title, url in references)
        + "</ul>",
        "Baseline tham chiếu MAE, điều chỉnh cho OrganAMNIST.\n\n"
        + "\n".join(f"- [{title}]({url})" for title, url in references),
    )
    css = """body{margin:0;background:#f5f7fa;color:#172b4d;font:16px/1.6 'Segoe UI',sans-serif}
main{max-width:1380px;margin:auto;padding:32px 20px}h1{font-size:32px;line-height:1.2}
section{background:white;padding:24px;margin:20px 0;border:1px solid #dce3eb;border-radius:12px}
h2{font-size:22px;margin:0 0 16px}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:12px;border-bottom:1px solid #dce3eb;text-align:left;
vertical-align:top;min-width:95px}
th{background:#edf3f8}.scroll{overflow-x:auto}img{width:100%;height:auto}pre{white-space:pre-wrap;
overflow-wrap:anywhere;font-size:13px}a{color:#086f83}
@media(max-width:600px){main{padding:16px 10px}
section{padding:16px}h1{font-size:26px}}"""
    page = "<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' "
    page += "content='width=device-width,initial-scale=1'><title>CT-MAE training report</title>"
    page += f"<style>{css}</style><main><h1>CT-MAE — cấu hình và kết quả thực nghiệm</h1>"
    page += "".join(blocks) + "</main></html>"
    (output / "training_report.html").write_text(page, "utf-8")
    (output / "training_report.md").write_text("\n\n".join(markdown) + "\n", "utf-8")
    return output / "training_report.html"
