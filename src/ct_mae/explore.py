"""Generate an offline visual report from the unchanged OrganAMNIST archives."""

import argparse
import csv
import json
from pathlib import Path

from ct_mae.collect import SPLITS, collect_dataset
from ct_mae.plots import write_plots


def write_report(data: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    templates = Path(__file__).parent
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = templates.joinpath("report.html").read_text(encoding="utf-8")
    for token, content in {
        "__DATA__": payload,
        "__STYLE__": templates.joinpath("report.css").read_text(encoding="utf-8"),
        "__SCRIPT__": templates.joinpath("report.js").read_text(encoding="utf-8"),
    }.items():
        html = html.replace(token, content)
    output.joinpath("explorer.html").write_text(html, encoding="utf-8")
    summary = {key: value for key, value in data.items() if key != "examples"}
    summary["example_indices"] = {
        split: [{"index": item["index"], "label": item["label"]} for item in items]
        for split, items in data["examples"].items()
    }
    output.joinpath("dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "samples",
        "dtype",
        "channels",
        "all_images_same_shape",
        "labeled_samples",
        "min",
        "max",
        "mean",
        "std",
        "p01",
        "p50",
        "p99",
        "zero_fraction",
        "max_fraction",
        "image_bytes",
        "array_bytes",
        "float32_array_bytes",
    ]
    with output.joinpath("split_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["resolution", "split", "shape", *fields])
        writer.writeheader()
        for size in data["sizes"]:
            for split in SPLITS:
                stats = data["resolutions"][str(size)]["splits"][split]
                writer.writerow(
                    {
                        "resolution": size,
                        "split": split,
                        "shape": str(stats["shape"]),
                        **{key: stats[key] for key in fields},
                    }
                )
    with output.joinpath("class_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["resolution", "split", "class_id", "class_name", "count", "fraction"])
        for size in data["sizes"]:
            for split in SPLITS:
                stats = data["resolutions"][str(size)]["splits"][split]
                for class_id, count in enumerate(stats["class_counts"]):
                    writer.writerow(
                        [
                            size,
                            split,
                            class_id,
                            data["metadata"]["labels"][class_id],
                            count,
                            count / stats["samples"],
                        ]
                    )
    write_plots(data, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--sizes", type=int, nargs="+", choices=[28, 64, 128, 224], default=[28, 64]
    )
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")
    data = collect_dataset(
        args.data_dir, sorted(set(args.sizes)), args.samples_per_class, args.seed
    )
    write_report(data, args.output)
    print(f"Report ready: {(args.output / 'explorer.html').resolve()}", flush=True)
    print("Preprocessing decisions remain open; original image values were preserved.", flush=True)


if __name__ == "__main__":
    main()
