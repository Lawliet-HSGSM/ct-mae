"""Offline Vietnamese report for measured classifier adaptation, including epoch-zero baselines."""

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

from ct_mae.collect import LABELS_VI, png_data_url
from ct_mae.config import write_json
from ct_mae.error_analysis import (
    classification_summary,
    paired_outcomes,
    read_predictions,
    select_errors,
)
from ct_mae.followup_report import figure_save, plt

NAMES = {'source': 'Nguồn · epoch 0', 'A': 'A · CE học tiếp', 'B': 'B · Sampler',
         'C': 'C · Trọng số organ', 'D': 'D · SupCon', 'E': 'E · Kết hợp'}
COLORS = ['#98a7b5', '#526782', '#d89922', '#19a07d', '#7953b8', '#df6a39']


def positive_count_bound(labels):
    """SupCon is bounded by mean(log positive count), with two views per base image."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return float(np.log(2 * counts[inverse] - 1).mean())


def collect_data(artifacts, archive):
    protocol = json.loads((artifacts / 'protocol.json').read_text('utf-8'))
    selection = json.loads((artifacts / 'selection_frozen.json').read_text('utf-8'))
    with np.load(archive, allow_pickle=False) as saved:
        images, labels = saved['test_images'], saved['test_labels'].reshape(-1)
        train_labels = saved['train_labels'].reshape(-1)
    verification = json.loads((artifacts / 'collection_verification.json').read_text('utf-8'))
    if verification['status'] != 'verified':
        raise ValueError('Generate the report only after independent collection verification')
    data = {'protocol': protocol, 'selection': selection, 'results': {}, 'paired': {},
            'environment': json.loads((artifacts / 'environment.json').read_text('utf-8')),
            'verification': verification}
    predictions = {}
    for scope in ('full', 'partial'):
        data['results'][scope], predictions[scope], data['paired'][scope] = {}, {}, {}
        for variant in NAMES:
            root = artifacts / scope / variant
            saved = json.loads((root / 'test.json').read_text('utf-8'))
            values = read_predictions(root / 'test_predictions.npz', labels)
            summary = classification_summary(labels, values['prediction'])
            for key in ('accuracy', 'macro_f1'):
                if not np.isclose(saved[key], summary[key], rtol=0, atol=1e-12):
                    raise ValueError('Report metrics disagree with saved model predictions')
            row = {**summary, 'nll': saved['nll'], 'ece_15': saved['ece_15'],
                   'checkpoint_sha256': saved['checkpoint_sha256'], 'epoch': 0}
            if variant != 'source':
                run = json.loads((root / 'run.json').read_text('utf-8'))
                row.update(epoch=run['best_epoch'], history=run['history'],
                           val_macro_f1=run['best_score'], wandb_url=run['wandb_url'],
                           parameters=run['parameters'],
                           final_val_macro_f1=run['history'][-1]['val']['macro_f1'],
                           final_supcon=run['history'][-1]['train']['supcon'],
                           train_seconds=sum(h['train']['seconds'] for h in run['history']),
                           updates=sum(h['train']['optimizer_steps'] for h in run['history']),
                           peak_vram=max(h['resource']['peak_vram_bytes'] for h in run['history']))
                if variant in ('D', 'E'):
                    with np.load(root / 'sampling_trace.npz', allow_pickle=False) as trace:
                        batches = trace['epoch_50'].reshape(-1, protocol['recipe']['batch_size'])
                    row['supcon_lower_bound'] = float(np.mean([
                        positive_count_bound(train_labels[batch]) for batch in batches]))
            else:
                row['val_macro_f1'] = protocol['initial_validation'][scope]['macro_f1']
            neighbors = root / 'neighbors.json'
            if neighbors.exists():
                row['neighbors'] = json.loads(neighbors.read_text('utf-8'))
            data['results'][scope][variant], predictions[scope][variant] = row, values
        for variant in 'ABCDE':
            data['paired'][scope][variant] = paired_outcomes(
                labels, predictions[scope]['source']['prediction'],
                predictions[scope][variant]['prediction'],
            )
    return data, images, labels, predictions


def plots(data, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    for row, scope in enumerate(('full', 'partial')):
        results = data['results'][scope]
        for axis, metric in zip(axes[row], ('accuracy', 'macro_f1'), strict=True):
            values = [results[v][metric] * 100 for v in NAMES]
            axis.bar(range(6), values, color=COLORS)
            axis.set_xticks(range(6), ['Nguồn', 'A', 'B', 'C', 'D', 'E'])
            label = 'Accuracy' if metric == 'accuracy' else 'macro-F1'
            axis.set(title=f'{scope.title()} · Test {label}', ylabel='%',
                     ylim=(max(0, min(values) - 3), min(100, max(values) + 2)))
            for i, value in enumerate(values):
                axis.text(i, value + .08, f'{value:.2f}', ha='center', fontsize=9)
    figure_save(fig, output, 'comparison')
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), layout='constrained')
    for axis, scope in zip(axes, ('full', 'partial'), strict=True):
        results = data['results'][scope]
        delta = np.array([[r['f1'] - b['f1'] for r, b in zip(
            results[v]['per_class'], results['source']['per_class'], strict=True)]
            for v in 'ABCDE']) * 100
        limit = max(1., float(np.abs(delta).max()))
        axis.imshow(delta, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        for r in range(5):
            for c in range(11):
                axis.text(c, r, f'{delta[r,c]:+.2f}', ha='center', va='center', fontsize=8)
        axis.set_xticks(range(11), LABELS_VI, rotation=35, ha='right', fontsize=8)
        axis.set_yticks(range(5), list('ABCDE'))
        axis.set_title(f'{scope.title()} · Δ F1 so với nguồn, điểm phần trăm')
    figure_save(fig, output, 'per_class_delta')
    for key in ('macro_f1', 'loss'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
        for axis, scope in zip(axes, ('full', 'partial'), strict=True):
            initial = data['protocol']['initial_validation'][scope][key]
            factor = 100 if key == 'macro_f1' else 1
            axis.axhline(initial * factor, color=COLORS[0], linestyle='--', label='Nguồn')
            for color, variant in zip(COLORS[1:], 'ABCDE', strict=True):
                history = data['results'][scope][variant]['history']
                axis.plot([0, *[h['epoch'] for h in history]],
                          [initial * factor, *[h['val'][key] * factor for h in history]],
                          color=color, label=variant)
            axis.set(title=f'{scope.title()} · validation {key}', xlabel='Epoch học tiếp')
            axis.legend()
            axis.grid(alpha=.2)
        figure_save(fig, output, f'validation_{key}')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for row, scope in enumerate(('full', 'partial')):
        for axis, field in zip(axes[row], ('ce', 'supcon', 'lambda'), strict=True):
            for color, variant in zip(COLORS[1:], 'ABCDE', strict=True):
                history = data['results'][scope][variant]['history']
                axis.plot([h['epoch'] for h in history], [h['train'][field] for h in history],
                          color=color, label=variant)
            axis.set(title=f'{scope.title()} · train {field}', xlabel='Epoch học tiếp')
            axis.legend()
            axis.grid(alpha=.2)
    figure_save(fig, output, 'loss_components')
    for scope in ('full', 'partial'):
        for variant in NAMES:
            counts = np.array(data['results'][scope][variant]['confusion'])
            percentages = counts / counts.sum(1, keepdims=True) * 100
            fig, axis = plt.subplots(figsize=(10, 9), layout='constrained')
            axis.imshow(percentages, cmap='Blues', vmin=0, vmax=100)
            for r in range(11):
                for c in range(11):
                    if counts[r, c]:
                        axis.text(c, r, f'{counts[r,c]}\n{percentages[r,c]:.1f}%',
                                  ha='center', va='center', fontsize=6,
                                  color='white' if percentages[r,c] > 55 else '#26435d')
            axis.set_xticks(range(11), LABELS_VI, rotation=60, ha='right', fontsize=8)
            axis.set_yticks(range(11), LABELS_VI, fontsize=8)
            axis.set(title=f'{scope.title()} · {NAMES[variant]} · số ảnh / % theo hàng',
                     xlabel='Dự đoán →', ylabel='Nhãn thật →')
            figure_save(fig, output, f'confusion_{scope}_{variant}')


def galleries(data, images, labels, predictions):
    result = {}
    for scope in ('full', 'partial'):
        result[scope] = {}
        base = predictions[scope]['source']
        for variant in dict.fromkeys(('E', data['selection'][scope]['variant'])):
            if variant == 'source':
                continue
            current = predictions[scope][variant]
            old_ok, new_ok = base['prediction'] == labels, current['prediction'] == labels
            groups = {
                'Được sửa': select_errors(labels, base['probability'], 6, new_ok),
                'Mới sai': select_errors(labels, current['probability'], 6, old_ok),
                'Cả hai sai': select_errors(labels, current['probability'], 6, ~old_ok),
            }
            controls = np.flatnonzero(old_ok & new_ok)
            groups['Cả hai đúng · ngẫu nhiên'] = np.random.default_rng(42).choice(
                controls, min(6, len(controls)), replace=False,
            )
            result[scope][variant] = {}
            for group, indices in groups.items():
                cases = []
                for index in indices:
                    models = {}
                    for name, values in (('Nguồn', base), (variant, current)):
                        predicted = int(values['prediction'][index])
                        confidence = float(values['probability'][index, predicted])
                        models[name] = {'prediction': predicted, 'confidence': confidence}
                    cases.append({'index': int(index), 'label': int(labels[index]),
                                  'image': png_data_url(images[index]), 'models': models})
                result[scope][variant][group] = cases
    return result


def result_rows(data, scope):
    results = data['results'][scope]
    base, control = results['source'], results['A']
    return [[NAMES[v], f'{r["val_macro_f1"]:.3%}', f'{r["accuracy"]:.2%}', f'{r["macro_f1"]:.2%}',
             f'{(r["macro_f1"]-base["macro_f1"])*100:+.2f}',
             f'{(r["macro_f1"]-control["macro_f1"])*100:+.2f}',
             str(r['epoch']), f'{r["nll"]:.4f}', f'{r["ece_15"]:.2%}']
            for v, r in results.items()]


def intervention_rows(data, scope):
    results = data['results'][scope]
    contrasts = [('B', 'A', 'Sampler'), ('C', 'B', 'Trọng số CE'),
                 ('D', 'B', 'SupCon'), ('E', 'C', 'Thêm SupCon khi có trọng số'),
                 ('E', 'D', 'Thêm trọng số khi có SupCon')]
    return [[f'{new}−{old}', meaning,
             f'{100*(results[new]["val_macro_f1"]-results[old]["val_macro_f1"]):+.3f}',
             f'{100*(results[new]["macro_f1"]-results[old]["macro_f1"]):+.2f}',
             f'{100*(results[new]["accuracy"]-results[old]["accuracy"]):+.2f}']
            for new, old, meaning in contrasts]


def table(headers, rows):
    def cells(values, tag):
        return ''.join(f'<{tag}>{html.escape(str(value))}</{tag}>' for value in values)
    return ('<div class="table-wrap"><table><thead><tr>' + cells(headers, 'th')
            + '</tr></thead><tbody>' + ''.join('<tr>' + cells(row, 'td') + '</tr>' for row in rows)
            + '</tbody></table></div>')


def write_report(data, output):
    body, markdown = [], ['# CT-MAE — Học tiếp checkpoint phân loại\n']
    notes = output / 'observations.json'
    observations = json.loads(notes.read_text('utf-8')) if notes.exists() else {}
    if observations and observations['selection'] != data['selection']:
        raise ValueError('Qualitative notes were written for a different selected experiment')

    def section(title):
        body.append(f'<section><h2>{html.escape(title)}</h2>')
        markdown.append(f'\n## {title}\n')

    def paragraph(text):
        body.append(f'<p>{html.escape(text)}</p>')
        markdown.append(text + '\n')

    def add_table(headers, rows):
        body.append(table(headers, rows))
        markdown.extend(['| ' + ' | '.join(headers) + ' |',
                         '| ' + ' | '.join('---' for _ in headers) + ' |',
                         *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows], ''])

    def figure(name, caption):
        encoded = base64.b64encode((output / 'figures' / f'{name}.png').read_bytes()).decode()
        body.append(f'<figure><img src="data:image/png;base64,{encoded}" '
                    f'alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption>'
                    '</figure>')
        markdown.append(f'![{caption}](figures/{name}.png)\n')

    section('Kết quả sau 50 epoch học tiếp')
    paragraph('Hai checkpoint nguồn dùng cùng 1.728 nhãn train. Full nguồn ở epoch 70, partial '
              'nguồn ở epoch 25. Mỗi nguồn có năm lượt học tiếp, seed 42; không train lại MAE.')
    paragraph('A: CE học tiếp; B: ghép batch theo cặp dễ nhầm; C: B + trọng số CE theo organ; '
              'D: B + SupCon; E: kết hợp cả ba. '
              'Các ảnh validation/test giữ nguyên split chính thức.')
    figure('comparison', 'Test Accuracy và macro-F1; checkpoint chọn bằng validation, gồm epoch 0. '
           'Trục Y thu hẹp để thấy chênh lệch.')
    headers = ['Nhánh', 'Val F1 macro', 'Test Acc', 'Test F1 macro', 'Δ F1 / nguồn (đpt)',
               'Δ F1 / A (đpt)', 'Epoch chọn', 'NLL', 'ECE15']
    for scope in ('full', 'partial'):
        paragraph(f'Nguồn {scope}:')
        add_table(headers, result_rows(data, scope))
        choice = data['selection'][scope]
        row, base = data['results'][scope][choice['variant']], data['results'][scope]['source']
        paragraph(f'Chọn bằng validation: {NAMES[choice["variant"]]}, epoch {choice["epoch"]}. '
                  f'Test Accuracy thay đổi {(row["accuracy"]-base["accuracy"])*100:+.2f} đpt; '
                  f'macro-F1 {(row["macro_f1"]-base["macro_f1"])*100:+.2f} đpt. '
                  'Đây là kết quả một seed, không phải ước lượng độ ổn định qua nhiều lượt.')
        kept = [v for v in 'ABCDE' if data['results'][scope][v]['epoch'] == 0]
        if kept:
            paragraph(f'{scope.title()} · {", ".join(kept)} đã train đủ 50 epoch nhưng chọn '
                      'epoch 0: không epoch học tiếp nào vượt F1 validation của nguồn. '
                      'Test giống nguồn là do dùng lại classifier nguồn, không phải '
                      'trọng số ở epoch 50 không thay đổi.')
    body.append('</section>')
    if observations.get('quantitative'):
        section('Những điểm rút ra từ lượt chạy này')
        for note in observations['quantitative']:
            paragraph(note)
        body.append('</section>')
    section('Thiết kế đối chứng và cách đọc')
    paragraph('A là CE học tiếp với shuffle thông thường. B chỉ đổi cách ghép batch. C dùng '
              'sampler B và trọng số theo organ. D dùng sampler B và CE+SupCon. E kết hợp cả hai. '
              'B−A kiểm tra sampler; C−B kiểm tra trọng số; D−B kiểm tra SupCon; '
              'E−C/E−D kiểm tra tác dụng kết hợp.')
    for scope in ('full', 'partial'):
        paragraph(f'{scope.title()} · so sánh từng thay đổi, '
                  'sau khi chọn epoch riêng bằng validation:')
        add_table(['So sánh', 'Thay đổi', 'Δ Val F1 (đpt)', 'Δ Test F1 (đpt)', 'Δ Test Acc (đpt)'],
                  intervention_rows(data, scope))
    paragraph('Nếu một nhánh chọn epoch 0, bảng phản ánh kết quả của quy tắc chọn checkpoint. '
              'Không suy ra mục tiêu loss đó không tác động trong lúc train. '
              'Một lượt seed 42 chưa đủ để quy chênh lệch nhỏ '
              'thành tác dụng ổn định của phương pháp.')
    paragraph('Mọi nhánh đều có hai affine view nhẹ mỗi ảnh: xoay ±5°, dịch tối đa 5%, bilinear, '
              'fill 0, không flip/crop. Mỗi index ảnh gốc xuất hiện đúng một lần mỗi epoch. '
              'A so với nguồn bao gồm việc học thêm, optimizer mới và cách học hai view; '
              'các nhánh học tiếp được đối chứng cùng điều kiện với A/B.')
    paragraph('Sampler ưu tiên gan–phổi phải, thận trái–lách, thận trái–thận phải, xác định từ '
              'validation nguồn. Ghép các cặp trong batch nhưng bảo toàn tần suất từng lớp. '
              'Trọng số organ lấy từ F1 validation nguồn, giữ cố định và chỉ áp dụng cho CE; '
              'không phải trọng số từng ảnh dựa trên chất lượng crop.')
    weight_rows = [[LABELS_VI[i], f'{data["protocol"]["class_weights"]["full"][i]:.3f}',
                    f'{data["protocol"]["class_weights"]["partial"][i]:.3f}'] for i in range(11)]
    add_table(['Organ', 'Trọng số CE · full', 'Trọng số CE · partial'], weight_rows)
    paragraph('Công thức: d=1−F1; raw=1+d/max(d); chia raw cho trung bình theo tỷ lệ lớp train. '
              'Trọng số trung bình bằng 1; tỷ lệ lớn nhất/nhỏ nhất tối đa 2. '
              'Nếu mọi F1=1 thì dùng 1.')
    paragraph('SupCon dùng projector 384→384→128, ReLU, chuẩn hóa L2, temperature 0.1; '
              'lambda tăng 0→0.05 trong năm epoch. Positive theo nhãn train. Pool so sánh '
              'chỉ trong micro-batch, không gộp qua tích lũy gradient. Inference bỏ projector.')
    paragraph('Mục tiêu học tiếp là weighted CE + lambda × SupCon (CE trọng số 1 ở A/B/D). '
              'SupCon lấy trung bình −log xác suất tương đồng của từng positive cùng nhãn '
              'so với mọi view khác ngoài chính anchor. Hai view của cùng ảnh cũng là positive. '
              'Đây là học có giám sát sau MAE; không dùng lại decoder hay MSE reconstruction.')
    paragraph('Validation được dùng thiết kế sampler/trọng số và chọn checkpoint. Không đưa '
              'ảnh validation/test vào train. Test chỉ được đánh giá sau khi đóng băng lựa chọn. '
              'Test đã được xem ở các vòng trước nên kết quả vẫn mang tính khám phá.')
    body.append('</section>')
    section('Organ và kiểu nhầm nào thay đổi?')
    figure('per_class_delta', 'F1 từng organ so với checkpoint nguồn, đơn vị điểm phần trăm.')
    for scope in ('full', 'partial'):
        rows = []
        results = data['results'][scope]
        chosen = data['selection'][scope]['variant']
        paragraph(f'{scope.title()} · precision/recall/F1 của nguồn và lựa chọn validation '
                  f'({NAMES[chosen]}):')
        organ_rows = []
        for before, after in zip(results['source']['per_class'], results[chosen]['per_class'],
                                 strict=True):
            organ_rows.append([LABELS_VI[before['label']], before['support'],
                               f'{before["f1"]:.2%}', f'{after["precision"]:.2%}',
                               f'{after["recall"]:.2%}', f'{after["f1"]:.2%}',
                               f'{100*(after["f1"]-before["f1"]):+.2f}'])
        add_table(['Organ', 'Ảnh test', 'F1 nguồn', 'Precision chọn', 'Recall chọn', 'F1 chọn',
                   'Δ F1 (đpt)'], organ_rows)
        for variant in 'ABCDE':
            source = np.array(results['source']['confusion'])
            current = np.array(results[variant]['confusion'])
            for a, b in data['protocol']['pairs']:
                for i, j in ((a, b), (b, a)):
                    rows.append([variant, LABELS_VI[i], LABELS_VI[j], str(source[i, j]),
                                 str(current[i, j]), f'{current[i,j]-source[i,j]:+d}'])
        body.append(f'<details><summary>{scope.title()} · các hướng nhầm ưu tiên</summary>')
        add_table(['Nhánh', 'Thật', 'Dự đoán', 'Nguồn', 'Sau học tiếp', 'Δ ảnh'], rows)
        body.append('</details>')
        for variant in NAMES:
            body.append(f'<details><summary>{scope.title()} · {html.escape(NAMES[variant])}'
                        ' · confusion matrix</summary>')
            figure(f'confusion_{scope}_{variant}',
                   'Hàng là nhãn thật, cột là dự đoán; % theo hàng.')
            add_table(['Organ', 'Precision', 'Recall', 'F1'], [
                [LABELS_VI[r['label']], f'{r["precision"]:.2%}', f'{r["recall"]:.2%}',
                 f'{r["f1"]:.2%}'] for r in results[variant]['per_class']])
            body.append('</details>')
    body.append('</section>')
    section('Đường học, tài nguyên và biểu diễn encoder')
    for name, caption in (('validation_macro_f1', 'Validation macro-F1 qua epoch học tiếp.'),
                          ('validation_loss', 'Validation CE sạch, cùng điều kiện đo.'),
                          ('loss_components', 'Các thành phần train loss và lambda; không xếp hạng '
                           'mô hình bằng tổng loss của những mục tiêu khác nhau.')):
        figure(name, caption)
    rows = []
    for scope in ('full', 'partial'):
        for variant in 'ABCDE':
            r = data['results'][scope][variant]
            rows.append([scope, variant, r['updates'], f'{r["train_seconds"]/60:.1f}',
                         f'{r["peak_vram"]/1024**3:.2f}', r['parameters']])
    add_table(['Nguồn', 'Nhánh', 'Updates thực', 'Phút train loop', 'Peak VRAM GiB',
               'Tham số học'], rows)
    paragraph('Thời gian trên là train loop, không gồm validation/checkpoint/setup. '
              'Recipe: AdamW, LR encoder/head 3e-5, projector ×10; warmup 5 epoch rồi cosine '
              'về 1e-6; betas (0.9,0.999), WD 0.05, không decay bias/LayerNorm; AMP FP16.')
    paragraph('Batch thực và tích lũy: ' + str(data['protocol']['recipe']['batch_size']) + ' × '
              + str(data['protocol']['recipe']['accumulation']) + ' ảnh gốc; hai view mỗi ảnh. '
              'Lưu trace index từng epoch và checkpoint đủ trạng thái resume. '
              'Partial chỉ mở ba block cuối, LayerNorm cuối và classifier; projector học ở D/E.')
    rows = [[scope, v, f'{r["final_val_macro_f1"]:.3%}', f'{r["final_supcon"]:.4f}',
             f'{r["supcon_lower_bound"]:.4f}' if 'supcon_lower_bound' in r else '—',
             r['epoch']] for scope, variants in data['results'].items()
            for v, r in variants.items() if v != 'source']
    add_table(['Nguồn', 'Nhánh', 'Val F1 epoch 50', 'SupCon train epoch 50',
               'Cận dưới theo positive', 'Epoch chọn'], rows)
    paragraph('SupCon bằng 0 ở A/B/C vì không dùng mục tiêu này. Loss SupCon của D/E đo ở '
              'projector; không so trực tiếp loss train với validation CE sạch. '
              'Bảng epoch 50 cho thấy trạng thái cuối dù checkpoint được giữ từ epoch sớm hơn.')
    paragraph('Nếu có n ảnh gốc cùng lớp trong micro-batch (tính cả ảnh đang xét), '
              'mỗi view có P=2n−1 '
              'positive. Loss của anchor không thể thấp hơn log(P), do phải chia xác suất '
              'cho các positive. Cận trong bảng là trung bình log(P) theo các batch thực ở '
              'epoch 50; không giả định mọi mô hình đều đạt được cận này. Vì vậy SupCon '
              'không nhất thiết tiến về 0 dù đã tối ưu tốt mục tiêu train.')
    neighbor_rows = [[scope, NAMES[v], f'{r["neighbors"]["agreement"]:.2%}']
                     for scope, rows in data['results'].items() for v, r in rows.items()
                     if 'neighbors' in r]
    add_table(['Nguồn', 'Checkpoint', '10 láng giềng cùng nhãn · validation'], neighbor_rows)
    paragraph('Cùng tập 1.100 ảnh validation, 100 mỗi organ; cosine trên feature encoder 384D '
              'trước projector. Đây là thống kê mô tả, không phải accuracy classifier. '
              'SupCon loss giảm ở projector chưa tự chứng minh encoder phân loại tốt hơn.')
    body.append('</section>')
    section('Ảnh được sửa và ảnh mới sai')
    for scope in ('full', 'partial'):
        rows = [[v, p['recovered'], p['regressed'], p['both_wrong'], p['both_correct'],
                 p['recovered']-p['regressed']]
                for v, p in data['paired'][scope].items()]
        paragraph(f'{scope.title()} · so sánh từng ảnh test với nguồn:')
        add_table(['Nhánh', 'Sai→đúng', 'Đúng→sai', 'Cùng sai', 'Cùng đúng', 'Sửa ròng'], rows)
    paragraph('Sửa ròng = số sai→đúng trừ số đúng→sai; chia 17.778 cho ra thay đổi Accuracy. '
              'Macro-F1 còn phụ thuộc precision/recall từng organ nên không suy từ con số này.')
    paragraph('Mỗi cặp là cùng ảnh test, cùng nhãn dữ liệu. Hiển thị pixel gốc 128×128, '
              'thang xám cố định 0–255, không làm mượt. Confidence là softmax chưa hiệu chuẩn. '
              'Ca lỗi chọn theo CE cao; mẫu đúng chọn ngẫu nhiên seed 42. '
              'Những gallery này không đại diện tỷ lệ xuất hiện của lỗi.')
    for scope, variants in data['galleries'].items():
        for variant, groups in variants.items():
            for group, cases in groups.items():
                body.append(f'<details><summary>{scope.title()} · {variant} · '
                            f'{html.escape(group)} · {len(cases)} ảnh</summary>'
                            '<div class="gallery">')
                markdown.append(f'\n### {scope} · {variant} · {group}\n')
                for case in cases:
                    caption = f'Test #{case["index"]} · Nhãn: {LABELS_VI[case["label"]]}'
                    body.append(f'<article class="case"><img src="{case["image"]}" '
                                f'alt="{html.escape(caption)}" width="128" height="128">'
                                f'<strong>{html.escape(caption)}</strong>')
                    lines = []
                    for name, value in case['models'].items():
                        state = 'good' if value['prediction'] == case['label'] else 'bad'
                        text = (f'{name}: {LABELS_VI[value["prediction"]]} · '
                                f'{value["confidence"]:.1%}')
                        body.append(f'<div class="{state}">{html.escape(text)}</div>')
                        lines.append(text)
                    body.append('</article>')
                    image_path = output / 'figures' / f'case_test_{case["index"]}.png'
                    image_path.write_bytes(base64.b64decode(case['image'].split(',')[1]))
                    markdown.append(f'![{caption}](figures/{image_path.name})\n'
                                    + ' · '.join(lines) + '\n')
                if not cases:
                    body.append('<p>Không có ảnh thuộc nhóm này ở checkpoint được chọn.</p>')
                    markdown.append('Không có ảnh thuộc nhóm này ở checkpoint được chọn.\n')
                body.append('</div></details>')
    paragraph('F1 hoặc Accuracy tăng không bảo đảm mọi hướng nhầm đều giảm. '
              'Cần đối chiếu ca được sửa với ca mới sai. Không kết luận vùng đen, texture, '
              'crop hoặc sai nhãn là nguyên nhân chỉ từ những ví dụ này; NPZ cũng thiếu scan ID '
              'để coi các crop tương tự nhau là các ca chụp độc lập.')
    for note in observations.get('notes', []):
        paragraph(note)
    body.append('</section>')
    section('Nguồn số liệu và giới hạn')
    env = data['environment']
    add_table(['Thuộc tính', 'Thực tế'], [
        ['GPU', env.get('gpu')], ['CUDA runtime', env.get('cuda_runtime')],
        ['Python', env.get('python')], ['Torch', env.get('versions', {}).get('torch')],
        ['Source commit train', env.get('git_commit')],
        ['Subset SHA-256', data['protocol']['subset_sha256']],
        *[[f'{scope} source SHA-256', digest] for scope, digest in
          data['protocol']['source_manifest']['checkpoints'].items()],
    ])
    paragraph('Mọi bảng được tính lại từ prediction đã lưu và kiểm tra alignment với nhãn '
              'test chính thức. Collection audit xác minh source/subset hashes, 50 epoch, '
              'sampling trace, lựa chọn epoch 0 hoặc earliest best '
              'và tensors đóng băng của partial.')
    for scope, rows in data['results'].items():
        for variant, row in rows.items():
            if row.get('wandb_url'):
                url = html.escape(row['wandb_url'], quote=True)
                body.append(f'<p><a href="{url}">{scope} · {NAMES[variant]} · W&amp;B</a></p>')
                markdown.append(f'[{scope} · {variant} · W&B]({row["wandb_url"]})\n')
    paragraph('Không có sweep LR/temperature/lambda và không có lặp seed trong lượt này. '
              'Giữ checkpoint nguồn nếu validation không có ứng viên tốt hơn. Test chỉ xác nhận '
              'kết quả của lựa chọn đã đóng băng; không chọn lại nhánh theo test. '
              'Cải thiện ngoài mẫu không được giả định trước.')
    references = [
        ('Supervised Contrastive Learning · Khosla et al. (2020)',
         'https://arxiv.org/abs/2004.11362'),
        ('Mã tham chiếu SupContrast · losses.py',
         'https://github.com/HobbitLong/SupContrast/blob/master/losses.py'),
        ('Masked Autoencoders Are Scalable Vision Learners · He et al. (2021)',
         'https://arxiv.org/abs/2111.06377'),
    ]
    paragraph('Tham khảo mục tiêu contrastive và encoder MAE; việc warm-start classifier rồi '
              'học CE+SupCon, lambda 0.05 và sampler ở đây là thiết kế điều chỉnh của project, '
              'không phải tái lập nguyên cấu hình paper SupCon.')
    for title, url in references:
        body.append(f'<p><a href="{url}">{html.escape(title)}</a></p>')
        markdown.append(f'[{title}]({url})\n')
    body.append('</section>')
    style = Path(__file__).with_name('followup.css').read_text('utf-8')
    style += ('.case img{display:block;width:128px;height:128px;margin:auto;'
              'image-rendering:pixelated}')
    page = ('<!doctype html><html lang="vi"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>CT-MAE · Học tiếp checkpoint</title><style>' + style + '</style><body><main>'
            '<header><p class="eyebrow">CT-MAE · ADAPTATION STUDY · 5% NHÃN</p>'
            '<h1>Học tiếp checkpoint phân loại</h1><p class="lead">Sampler, trọng số organ '
            'và supervised contrastive loss · Full & Partial · seed 42.</p></header>'
            + ''.join(body) + '</main></body></html>')
    (output / 'adaptation_report.html').write_text(page, 'utf-8')
    (output / 'adaptation_report.md').write_text('\n'.join(markdown), 'utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        data = json.loads((args.output / 'adaptation_comparison.json').read_text('utf-8'))
    else:
        data, images, labels, predictions = collect_data(args.artifacts, args.archive)
        plots(data, args.output)
        data['galleries'] = galleries(data, images, labels, predictions)
        write_json(args.output / 'adaptation_comparison.json', data)
    write_report(data, args.output)


if __name__ == '__main__':
    main()
