"""Show true-pancreas test crops alongside an existing misclassified reference."""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from ct_mae.collect import LABELS_VI, png_data_url
from ct_mae.download import file_md5, official_asset
from ct_mae.error_analysis import read_predictions

STAGES = {'scratch': 'Scratch', 'linear': 'Linear', 'finetune': 'Full', 'partial': 'Partial'}


def export_page(images, labels, predictions, selection, output):
    reference = selection['reference']
    pool = np.flatnonzero(labels == 9)
    encoded = {}
    for index in sorted(set(pool.tolist()) | {reference}):
        encoded[index] = png_data_url(images[index])
        decoded = Image.open(BytesIO(base64.b64decode(encoded[index].split(',')[1])))
        if not np.array_equal(np.asarray(decoded), images[index]):
            raise ValueError('Gallery PNG differs from original source pixels')
    selection['verified_source_pngs'] = len(encoded)

    def card(index):
        lines = []
        for stage, name in STAGES.items():
            predicted = int(predictions[stage]['prediction'][index])
            confidence = predictions[stage]['probability'][index, predicted]
            state = 'good' if predicted == labels[index] else 'bad'
            lines.append(f'<div class="{state}">{name}: {LABELS_VI[predicted]} · '
                         f'{confidence:.1%}</div>')
        return (f'<article class="case" data-index="{index}" data-label="{labels[index]}">'
                f'<img src="{encoded[index]}" width="128" height="128" loading="lazy" '
                f'alt="Test #{index}, nhãn dữ liệu {LABELS_VI[labels[index]]}">'
                f'<strong>#{index} · Nhãn: {LABELS_VI[labels[index]]}</strong>'
                + ''.join(lines) + '</article>')

    def gallery(indices):
        return '<div class="gallery">' + ''.join(card(int(i)) for i in indices) + '</div>'

    style = (Path(__file__).resolve().parents[1] / 'src/ct_mae/followup.css').read_text('utf-8')
    style += ('.case img{display:block;width:128px;height:128px;margin:auto;'
              'image-rendering:pixelated}.gallery{margin-top:20px;gap:20px}')
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CT-MAE · Ảnh tụy và tham chiếu #{reference}</title><style>{style}</style>
<body><main><header><p class="eyebrow">CT-MAE · ĐỐI CHIẾU ẢNH TEST · 5% NHÃN</p>
<h1>Ảnh tụy trong OrganAMNIST</h1><p class="lead">So sánh #{reference} với ảnh có
nhãn dữ liệu là tụy. Tập test có {len(pool):,} ảnh tụy.</p>
<p><a href="#random">Mẫu ngẫu nhiên</a> · <a href="#nearest">Gần theo bố cục</a> ·
<a href="#all">Toàn bộ ảnh tụy</a> · <a href="followup_report.html">Báo cáo follow-up</a></p>
<p>Ảnh giữ nguyên pixel 128×128, thang xám 0–255 và hiển thị 1:1. Dự đoán lấy từ
bốn checkpoint của lượt 5% nhãn; phần trăm là softmax chưa hiệu chuẩn.</p></header>
<section id="reference"><h2>Ảnh tham chiếu #{reference}</h2>
{gallery([reference])}<p>Nhãn của ảnh này trong dataset là
<strong>{LABELS_VI[labels[reference]]}</strong>. Các ảnh đối chiếu bên dưới có nhãn
tụy theo dataset. Việc giống nhau khi nhìn chưa xác nhận được đúng/sai của nhãn.</p></section>
<section id="random"><h2>15 ảnh tụy chọn ngẫu nhiên</h2>
<p>Seed 42; chọn trong toàn bộ lớp tụy, không lọc theo kết quả dự đoán.</p>
{gallery(selection['random_indices'])}
<p><a href="figures/pancreas_random_{reference}.png">Mở lưới ảnh lớn kèm tham chiếu</a></p>
</section>
<section id="nearest"><h2>15 ảnh tụy gần nhất theo bố cục pixel</h2>
<p>Xếp theo MSE sau khi lấy trung bình pixel xuống 32×32 để tìm ảnh. Ảnh hiển thị
vẫn là bản 128×128 gốc. Khoảng cách này đo bố cục sáng–tối, chưa xác nhận tương đồng
giải phẫu và không phải khoảng cách embedding của encoder.</p>
{gallery(selection['nearest_indices'][:15])}
<p><a href="figures/pancreas_nearest_{reference}.png">Mở lưới ảnh lớn kèm tham chiếu</a></p>
</section>
<section id="all"><h2>Toàn bộ ảnh tụy trong test</h2>
<details><summary>Mở {len(pool):,} ảnh · theo thứ tự index test chính thức</summary>
{gallery(pool)}</details><p>Ảnh được chọn theo nhãn dataset; bao gồm cả ảnh các model
đoán đúng và đoán sai. Các crop tương tự nhau có thể cùng CT scan; NPZ thiếu scan ID.</p>
<p><a href="pancreas_review.json">Danh sách index và cách chọn mẫu</a></p></section>
</main></body></html>'''
    (output / 'pancreas_review.html').write_text(page, encoding='utf-8')


def draw_sheet(images, labels, predictions, indices, reference, title, target):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.6), layout='constrained')
    for axis, index in zip(axes.flat, [reference, *indices], strict=True):
        axis.imshow(images[index], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axis.set_axis_off()
        predicted = int(predictions['finetune']['prediction'][index])
        confidence = predictions['finetune']['probability'][index, predicted]
        prefix = 'THAM CHIẾU · ' if index == reference else ''
        axis.set_title(f'{prefix}#{index} · Nhãn: {LABELS_VI[labels[index]]}\n'
                       f'Full: {LABELS_VI[predicted]} · {confidence:.1%}', fontsize=9,
                       color='#b91c1c' if predicted != labels[index] else '#134e4a')
    fig.suptitle(title + '\nPixel gốc 128×128 · thang xám cố định 0–255 · '
                 'dự đoán từ lượt 5% nhãn', fontsize=13)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=int, default=10267)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'reports/followup'
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    archive = root / 'data/organamnist_128.npz'
    digest = file_md5(archive)
    if digest != official_asset(128)['md5']:
        raise ValueError('Official archive checksum mismatch')
    with np.load(archive, allow_pickle=False) as saved:
        images, labels = saved['test_images'], saved['test_labels'].reshape(-1)
    reference = args.reference
    if not 0 <= reference < len(labels):
        raise ValueError('Reference must be an official test index')
    if images.dtype != np.uint8 or images.shape != (len(labels), 128, 128):
        raise ValueError('Expected original grayscale uint8 test images')
    predictions = {}
    for stage in STAGES:
        base = root / ('reports/followup/artifacts' if stage == 'partial'
                       else 'reports/baseline')
        predictions[stage] = read_predictions(base / stage / 'test_predictions.npz', labels)
    pool = np.flatnonzero(labels == 9)
    pool = pool[pool != reference]
    assert LABELS_VI[9] == 'Tụy'
    features = images.reshape(-1, 32, 4, 32, 4).mean(axis=(2, 4), dtype=np.float32) / 255
    distance = np.square(features[pool] - features[reference]).mean(axis=(1, 2))
    nearest = pool[np.argsort(distance, kind='stable')]
    random = np.random.default_rng(42).choice(pool, 15, replace=False)
    selection = {
        'reference': reference, 'reference_label': int(labels[reference]),
        'organ_label': 9, 'test_organ_count': int(np.count_nonzero(labels == 9)), 'random_seed': 42,
        'random_indices': random.tolist(), 'nearest_indices': nearest.tolist(),
        'nearest_mse': np.sort(distance, kind='stable').tolist(),
        'matching': 'MSE of area-averaged 32x32 grayscale pixels /255; '
                    'no anatomy verification, no prediction-outcome filtering',
        'dataset_md5': digest, 'budget': 'Original 5%-label checkpoints, seed 42',
    }
    for name, indices, title in (
        ('random', random, 'Ảnh #10267 và 15 ảnh có nhãn tụy · chọn ngẫu nhiên, seed 42'),
        ('nearest', nearest[:15], 'Ảnh #10267 và 15 ảnh có nhãn tụy gần nhất theo bố cục pixel'),
    ):
        draw_sheet(images, labels, predictions, indices, reference, title,
                   figures / f'pancreas_{name}_{reference}.png')
    export_page(images, labels, predictions, selection, output)
    (output / 'pancreas_review.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(json.dumps({'reference': reference, 'pancreas_test_images': selection['test_organ_count'],
                      'random': random.tolist(), 'nearest': nearest[:15].tolist()},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
