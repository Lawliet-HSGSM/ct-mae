"""Static figures and offline presentation for the split diagnostic."""

import base64
import html
import json
from pathlib import Path

import numpy as np

from ct_mae.collect import LABELS_VI
from ct_mae.config import write_json
from ct_mae.followup_report import figure_save, plt
from ct_mae.split_review import METRICS

COLORS = ['#3979b7', '#8464b3', '#18887a', '#df8442']
GROUPS = [('train5', 'Train 5% gốc'), ('train_aug', 'Train 5% augment'),
          ('val', 'Validation'), ('test', 'Test')]


def make_figures(data, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), layout='constrained')
    for ax, (model, gap) in zip(axes, data['gaps'].items(), strict=True):
        for offset, split, color in [(-.18, 'validation', COLORS[2]), (.18, 'test', COLORS[3])]:
            values = [r['recall'] * 100 for r in gap[split]['per_class']]
            ax.bar(np.arange(11) + offset, values, width=.36, label=split, color=color)
        ax.set_xticks(range(11), LABELS_VI, rotation=30, ha='right', fontsize=9)
        ax.set(title=f'{model.title()} nguồn 5% · tỷ lệ đoán đúng trong từng organ (recall)',
               ylabel='%', ylim=(0, 105))
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    figure_save(fig, output, 'recall_gap')
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    for i, (split, title) in enumerate([('train', 'Toàn train'), ('train5', '5% có nhãn'),
                                       ('val', 'Validation'), ('test', 'Test')]):
        counts = np.array(data['class_counts'][split])
        ax.bar(np.arange(11) + (i - 1.5) * .2, counts / counts.sum() * 100, width=.2,
               color=COLORS[i], label=f'{title} · n={counts.sum():,}')
    ax.set_xticks(range(11), LABELS_VI, rotation=30, ha='right', fontsize=9)
    ax.set(ylabel='% ảnh trong tập', title='Tỷ trọng organ trong từng tập')
    ax.legend(ncol=2)
    figure_save(fig, output, 'class_mix')
    for organ, title in enumerate(LABELS_VI):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
        for ax, (metric, name) in zip(axes.flat, METRICS.items(), strict=True):
            boxes = []
            for split, label in GROUPS:
                q = data['summaries'][split][str(organ)][metric]['q']
                boxes.append(dict(label=label, whislo=q[0], q1=q[1], med=q[2], q3=q[3],
                                  whishi=q[4], fliers=[]))
            artists = ax.bxp(boxes, showfliers=False, patch_artist=True)
            for patch, color in zip(artists['boxes'], COLORS, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(.6)
            ax.set_title(name)
            ax.tick_params(axis='x', labelsize=8)
            ax.grid(axis='y', alpha=.2)
        fig.suptitle(f'{title} · cùng nhãn organ, toàn bộ ảnh trong mỗi nhóm\n'
                     'Hộp: 25–75%, vạch: median, râu: 10–90%; không phải điểm chất lượng')
        figure_save(fig, output, f'pixels_{organ}')


def write_review(data, output):
    make_figures(data, output)
    data['figures'] = {p.stem: 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
                       for p in (output / 'figures').glob('*.png')}
    rows, insights, markdown = [], [], ['# Train, augmentation, validation và test\n']
    for model, gap in data['gaps'].items():
        values = [model.title(), f'{gap["validation"]["accuracy"]:.2%}',
                  f'{gap["test"]["accuracy"]:.2%}',
                  f'{gap["class_mix_effect"]*100:+.2f}',
                  f'{gap["within_class_effect"]*100:+.2f}']
        cells = ''.join(f'<td>{html.escape(value)}</td>' for value in values)
        rows.append('<tr>' + cells + '</tr>')
        insights.append(f'<p>{html.escape(model.title())}: đổi tỷ trọng lớp chỉ đóng góp '
                        f'{values[3]} điểm phần trăm, còn thay đổi recall trong từng lớp '
                        f'đóng góp {values[4]} điểm. Khoảng cách chủ yếu nằm ở việc model '
                        'đoán khác đi trong cùng organ, chưa xác định yếu tố ảnh gây ra lỗi.</p>')
        markdown.append(f'{model}: validation {values[1]}, test {values[2]}; '
                        f'phần chênh do tỷ trọng lớp {values[3]} đpt, '
                        f'phần do recall trong từng lớp {values[4]} đpt.\n')
    template = Path(__file__).with_name('split_review.html').read_text('utf-8')
    css = Path(__file__).with_name('followup.css').read_text('utf-8')
    css += Path(__file__).with_name('split_review.css').read_text('utf-8')
    script = Path(__file__).with_name('split_review.js').read_text('utf-8')
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    page = template.replace('__STYLE__', css).replace('__ROWS__', ''.join(rows))
    page = page.replace('__INSIGHTS__', ''.join(insights))
    kidney = data['gaps']['full']
    dark = data['conditional_errors']['4']['dark']
    low, high = dark[1], dark[3]
    note = (f'Thận trái · Full: recall validation '
            f'{kidney["validation"]["per_class"][4]["recall"]:.1%}, '
            f'test {kidney["test"]["per_class"][4]["recall"]:.1%}. '
            f'Trong nhóm có nhiều pixel tối nhất theo mốc train, test có '
            f'{high["test"]["samples"]} ảnh với tỷ lệ lỗi '
            f'{high["test"]["models"]["full"]:.1%}, còn validation chỉ có '
            f'{high["val"]["samples"]} ảnh trong nhóm này. '
            f'Nhóm gần như không có pixel tối trên test vẫn lỗi '
            f'{low["test"]["models"]["full"]:.1%} trong {low["test"]["samples"]} ảnh. '
            'Vùng tối có liên hệ với lỗi ở nhóm này nhưng chưa giải thích hết khoảng cách, '
            'và chưa được xác nhận là nguyên nhân.')
    page = page.replace('__KIDNEY__', note)
    page = page.replace('__DATA__', encoded).replace('__SCRIPT__', script)
    (output / 'split_review.html').write_text(page, 'utf-8')
    write_json(output / 'split_review.json', data)
    markdown.extend([
        note + '\n',
        '## Cách đọc\n',
        'Trang HTML có bốn cột: train gốc, cùng ảnh sau affine, validation và test. '
        'Chọn organ, nguồn train, model, nhóm đúng/sai, trang mẫu và cách phóng ảnh. '
        'Mẫu ngẫu nhiên seed42; nhóm lỗi là lấy ngẫu nhiên trong toàn bộ lỗi cùng organ, '
        'không chỉ lấy các lỗi tự tin nhất.\n',
        '![Recall từng organ](figures/recall_gap.png)\n',
        '![Tỷ trọng lớp](figures/class_mix.png)\n',
        '## Giới hạn\n',
        'Chênh lệch accuracy được tách theo công thức kế toán, không phải xác định nguyên nhân. '
        'Đặc tính pixel không phải mask giải phẫu hay thước đo chất lượng CT đã xác thực. '
        'Vùng tối không đồng nghĩa với phổi. Chỉ train/augment là cùng ảnh; val/test '
        'không phải cặp cùng ca chụp. NPZ không chứa scan ID để kiểm chứng từng cặp.\n',
        'Augmentation dùng đúng TwinViewDataset: affine ±5°, dịch tối đa5%, bilinear, '
        'fill0, không flip/crop; seed42, epoch0. Thống kê augment lấy view0 của mỗi '
        'ảnh trong1728 nhãn gốc; minh họa có thêm view1. PNG augment lượng tử hóa '
        'về uint8 để xem, còn tensor train và phép đo dùng float32.\n',
        'Thông tin cách chia theo ca CT trong HTML được đối chiếu với '
        '[MedMNIST v2](https://www.nature.com/articles/s41597-022-01721-8). '
        'Không coi mọi crop là một ca độc lập.\n',
        'Các bản128 giữ split/index theo [MedMNIST+ chính thức]'
        '(https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md). '
        'Chưa đủ bằng chứng để kết luận một yếu tố ảnh cụ thể gây khoảng cách val–test.\n',
    ])
    (output / 'split_review.md').write_text('\n'.join(markdown), 'utf-8')
