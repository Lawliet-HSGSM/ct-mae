"""A compact offline narrative for the two measured label budgets."""

import base64
import html
import json
from pathlib import Path

import numpy as np

from ct_mae.followup_report import NAMES, SHORT, STAGES, embedded_figure, table


def percent(value):
    return f'{value * 100:.2f}%'


def write_report(data, output):
    budgets = data['budgets']
    methods = {key: budgets[key]['methods'] for key in ('5', '10')}
    paragraphs, markdown, figure_payload = [], [], {}

    def prose(text):
        paragraphs.append(f'<p>{html.escape(text)}</p>')
        markdown.append(text + '\n')

    def section(identifier, title):
        paragraphs.append(f'<section id="{identifier}"><h2>{html.escape(title)}</h2>')
        markdown.append(f'\n## {title}\n')

    def end():
        paragraphs.append('</section>')

    def figure(name, caption):
        paragraphs.append(embedded_figure(output, name, caption))
        if (output / f'figures/{name}.png').exists():
            markdown.append(f'![{caption}](figures/{name}.png)\n')

    def report_table(headers, rows):
        paragraphs.append(table(headers, rows))
        markdown.append('| ' + ' | '.join(headers) + ' |\n|' + '---|' * len(headers) + '\n'
                        + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
                        + '\n')

    markdown.extend(['# CT-MAE — So sánh 5% và 10% nhãn',
                     '\nKết quả một seed 42, mang tính khám phá.'])
    section('results', 'Kết quả đo được ở hai ngân sách nhãn')
    prose(f'Mốc 5% dùng {budgets["5"]["count"]:,} ảnh có nhãn; mốc 10% dùng '
          f'{budgets["10"]["count"]:,} ảnh. Mỗi nhánh chạy đủ 100 epoch, chọn checkpoint '
          'bằng validation macro-F1 rồi đánh giá test. Không chọn checkpoint theo test.')
    overview = []
    for stage in STAGES:
        a, b = methods['5'][stage], methods['10'][stage]
        overview.append([NAMES[stage], percent(a['accuracy']), percent(b['accuracy']),
                         f'{(b["accuracy"] - a["accuracy"]) * 100:+.2f}',
                         percent(a['macro_f1']), percent(b['macro_f1']),
                         f'{(b["macro_f1"] - a["macro_f1"]) * 100:+.2f}', b['best_epoch']])
    headers = ['Nhánh', 'Acc 5%', 'Acc 10%', 'Δ Acc (đpt)', 'Macro-F1 5%', 'Macro-F1 10%',
               'Δ macro-F1 (đpt)', 'Epoch chọn 10%']
    report_table(headers, overview)
    prose('Accuracy là tỷ lệ ảnh đoán đúng. Macro-F1 lấy trung bình F1 của 11 lớp, '
          'mỗi lớp có trọng số bằng nhau. Δ là thay đổi 10% − 5%, đơn vị điểm phần trăm. '
          'Bảng chỉ có một lượt seed 42; chênh lệch nhỏ chưa thể hiện độ ổn định qua seed.')
    figure('label_comparison', 'Hai điểm ngân sách đã đo; không ngoại suy các tỷ lệ chưa chạy.')
    for stage in STAGES:
        a, b = methods['5'][stage], methods['10'][stage]
        paired = data['paired'][stage]
        prose(f'{NAMES[stage]}: Accuracy thay đổi {(b["accuracy"] - a["accuracy"]) * 100:+.2f} '
              f'đpt và macro-F1 {(b["macro_f1"] - a["macro_f1"]) * 100:+.2f} đpt. '
              f'Mốc 10% sửa được {paired["recovered"]:,} ảnh mà 5% sai, đồng thời làm sai '
              f'{paired["regressed"]:,} ảnh mà 5% đúng; {paired["both_wrong"]:,} ảnh cả hai sai.')
    end()

    section('protocol', 'Thiết kế đối chứng và điều cần lưu ý')
    prose('Giữ nguyên OrganAMNIST 128×128, grayscale, 11 lớp; train/validation/test chính '
          'thức 34.561/6.491/17.778 ảnh. Chọn 10% bằng cách giữ toàn bộ tập 5% cũ, rồi '
          'lấy thêm ảnh theo phân bổ tỷ lệ lớp, không lấy trùng. Cả bốn nhánh ở mốc 10% '
          'dùng đúng cùng danh sách index, seed 42. “5%/10%” chỉ tính nhãn train; vẫn '
          'dùng toàn bộ nhãn validation để chọn checkpoint.')
    prose('Các con số 1.728 và 3.456 là số crop có nhãn, không phải số bệnh nhân. '
          'NPZ thiếu scan ID, nên chưa biết thêm ảnh đã mở rộng độ bao phủ ca chụp '
          'bao nhiêu. Những crop gần nhau từ cùng scan có thể tương quan mạnh.')
    class_rows = [[data['labels_vi'][i], budgets['5']['manifest']['labeled_class_counts'][i],
                   budgets['10']['manifest']['labeled_class_counts'][i], data['counts']['train'][i],
                   data['counts']['val'][i], data['counts']['test'][i]] for i in range(11)]
    report_table(['Organ', 'Nhãn 5%', 'Nhãn 10%', 'Train đầy đủ', 'Val', 'Test'], class_rows)
    prose('MAE đã học toàn bộ ảnh train không nhãn từ lượt trước; thí nghiệm này tái sử '
          'dụng chính checkpoint đó. Encoder 12 block, chiều 384, 6 heads, patch 8×8. '
          'Các classifier dùng ảnh đầy đủ → mean patch pooling → Linear(384,11). '
          'Scratch khởi tạo ngẫu nhiên; linear probing chỉ train head; full mở toàn '
          'encoder; partial mở 3 block cuối, LayerNorm cuối và head. Ba nhánh SSL '
          'dùng cùng một bộ trọng số MAE ở cả hai ngân sách.')
    recipe_rows = []
    for stage in STAGES:
        method = methods['10'][stage]
        recipe = method['recipe']
        recipe_rows.append([NAMES[stage], recipe['epochs'], recipe['lr'],
                            f'{recipe["batch_size"]} × {recipe["accumulation"]}',
                            recipe['weight_decay'], method['parameters']['trainable']])
    report_table(['Nhánh 10%', 'Epoch', 'Peak LR thực', 'Batch × tích lũy',
                  'Weight decay', 'Tham số train'], recipe_rows)
    prose('Giữ AdamW, betas (0.9,0.999), epsilon 1e-8; warmup 5 epoch rồi cosine về '
          '1e-6. Không decay bias/LayerNorm. AMP FP16; stochastic depth 0.1 khi '
          'train encoder, phần đóng băng ở eval. Train dùng affine nhẹ ±5°, dịch '
          'tối đa 5%, bilinear, fill 0; val/test chỉ chia pixel 255, không augmentation.')
    prose('Giữ 100 epoch ở cả hai mốc làm số lượt ảnh và optimizer updates tăng cùng '
          'số nhãn. Với batch hiệu dụng 64, kế hoạch là 27 updates/epoch ở 5% và 54 '
          'ở 10%. Warmup/cosine cũng diễn ra trên nhiều updates hơn ở 10%. Đây là '
          'so sánh ngân sách nhãn với cùng số epoch, chưa cô lập tác dụng của nhãn '
          'khỏi tác dụng của thêm bước tối ưu. Không kết luận toàn bộ mức tăng là do '
          'nhãn; muốn kiểm soát chặt yếu tố này cần một thí nghiệm cố định updates/LR theo step.')
    cost_rows = []
    for stage in STAGES:
        a, b = methods['5'][stage], methods['10'][stage]
        cost_rows.append([NAMES[stage], a['updates'], b['updates'],
                          f'{a["seconds"] / 60:.1f}', f'{b["seconds"] / 60:.1f}',
                          f'{b["peak_vram_bytes"] / 1024**3:.2f}'])
    report_table(['Nhánh', 'Updates 5%', 'Updates 10%', 'Phút 5%', 'Phút 10%',
                  'Peak VRAM 10% (GiB)'], cost_rows)
    prose('Updates là số bước thực hiện thành công từ log. Thời gian giai đoạn gồm '
          'train, validation và ghi checkpoint/log, không gồm tải dữ liệu. Các phiên '
          'Kaggle riêng có thể khác tốc độ dù cùng loại GPU. Checkpoint lưu đủ trạng '
          'thái optimizer, AMP scaler, scheduler và RNG để phục vụ resume.')
    prose('Bảng thời gian chỉ tính downstream. Chi phí 200 epoch MAE đã chạy trước '
          'đó không được tính lại vào từng hàng. So với scratch, các nhánh SSL vẫn '
          'có thêm quyền dùng toàn bộ ảnh train không nhãn và chi phí pretraining; '
          'đây là so sánh hiệu quả nhãn, không phải so sánh cùng tổng chi phí tính toán.')
    end()

    section('classes', 'Organ nào được lợi khi thêm nhãn?')
    figure('per_class_comparison', 'F1 từng organ: nét đứt 5%, nét liền 10%.')
    figure('per_class_delta', 'Màu biểu diễn Δ F1, đơn vị điểm phần trăm.')
    for stage in STAGES:
        a, b = methods['5'][stage], methods['10'][stage]
        delta = np.array([b['per_class'][i]['f1'] - a['per_class'][i]['f1'] for i in range(11)])
        best = int(delta.argmax())
        decreases = [f'{data["labels_vi"][i]} ({delta[i] * 100:+.2f} đpt)'
                     for i in np.argsort(delta) if delta[i] < 0]
        direction = 'tăng nhiều nhất' if delta[best] > 0 else 'thay đổi cao nhất'
        prose(f'{NAMES[stage]}: {direction} ở {data["labels_vi"][best]} '
              f'({delta[best] * 100:+.2f} đpt). ' + ('Các lớp giảm F1: ' + '; '.join(decreases)
              + '.' if decreases else 'Không có lớp giảm F1 trong lượt đo này.'))
    focused = []
    for stage in STAGES:
        for source, target in ((3, 7), (10, 7), (4, 10), (4, 5), (5, 4), (7, 8), (8, 7)):
            old = methods['5'][stage]['confusion'][source][target]
            new = methods['10'][stage]['confusion'][source][target]
            focused.append([SHORT[stage], data['labels_vi'][source], data['labels_vi'][target],
                            old, new, new - old])
    paragraphs.append('<details><summary>Các hướng nhầm đã theo dõi từ báo cáo trước</summary>')
    markdown.append('\n### Các hướng nhầm đã theo dõi từ báo cáo trước\n')
    report_table(['Nhánh', 'Thật', 'Dự đoán', 'Số ảnh 5%', 'Số ảnh 10%', 'Δ ảnh'], focused)
    paragraphs.append('</details>')
    prose('Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán. Ô đường chéo '
          'là đúng; ô ngoài đường chéo cho biết nhầm từ đâu sang đâu. Hình ghi cả '
          'số ảnh và % theo hàng, để so các lớp có số ảnh khác nhau.')
    paragraphs.append('''<div class="controls"><label>Nhánh<select id="matrix-stage"></select>
</label><label>Ngân sách<select id="matrix-budget"><option value="5">5% nhãn</option>
<option value="10" selected>10% nhãn</option></select></label></div>
<figure><img id="matrix-image" alt="Confusion matrix đã chọn"><figcaption id="matrix-caption">
</figcaption></figure>''')
    for budget in ('5', '10'):
        for stage in STAGES:
            path = output / f'figures/confusion_{stage}_{budget}.png'
            figure_payload[f'{stage}_{budget}'] = 'data:image/png;base64,' + (
                base64.b64encode(path.read_bytes()).decode('ascii'))
            markdown.append(f'![{NAMES[stage]} · {budget}% nhãn]'
                            f'(figures/confusion_{stage}_{budget}.png)\n')
    end()

    section('learning', 'Đường học và chọn checkpoint')
    figure('validation_epochs', 'Macro-F1 validation theo epoch; không dùng test để chọn model.')
    figure('validation_updates', 'Cùng trục updates để quan sát; lịch LR theo step vẫn khác nhau.')
    figure('training_loss', 'Train cross-entropy có augmentation và stochastic depth.')
    figure('validation_loss', 'Validation cross-entropy sau epoch, eval và ảnh sạch.')
    prose('Checkpoint ở epoch sớm vẫn có thể được chọn dù train chạy đủ 100 epoch. '
          'Chọn macro-F1 validation cao nhất; nếu bằng điểm giữ checkpoint xuất hiện '
          'trước. Val loss thấp hơn train loss không tự động do val ít ảnh: cách đo, '
          'augmentation, độ khó và nguồn scan cũng khác. Nguồn scan theo split chính '
          'thức là 115/16/70; NPZ thiếu scan ID để phân tích độ dao động theo ca.')
    end()

    section('cases', 'Ảnh nào được sửa và ảnh nào bị giảm hiệu quả?')
    prose('Mỗi cặp dùng đúng cùng ảnh test và nhãn thật. Chọn tối đa 6 ảnh được sửa '
          'hoặc bị giảm hiệu quả theo −log P(nhãn thật), để xem những thay đổi nổi '
          'bật. Chúng không đại diện tỷ lệ xuất hiện của tất cả lỗi. Hiển thị pixel '
          'gốc 128×128, grayscale cố định 0–255. Confidence là softmax chưa hiệu chuẩn; '
          'ảnh sai vẫn có thể được báo gần 100%.')
    observations = output / 'observations.json'
    if observations.exists():
        for note in json.loads(observations.read_text('utf-8')):
            prose(note)
    for stage in STAGES:
        paragraphs.append(f'<details><summary>{html.escape(NAMES[stage])}: ảnh thay đổi</summary>')
        for name, caption in (('recovered', '5% sai → 10% đúng'),
                               ('regressed', '5% đúng → 10% sai')):
            figure(f'cases_{stage}_{name}', f'{SHORT[stage]} · {caption}')
        paragraphs.append('</details>')
    end()

    section('interpretation', 'Có thể kết luận gì từ lượt này?')
    old_difference = (methods['5']['finetune']['accuracy']
                      - methods['5']['partial']['accuracy']) * 100
    new_difference = (methods['10']['finetune']['accuracy']
                      - methods['10']['partial']['accuracy']) * 100
    prose(f'Chênh lệch Accuracy Full − Partial là {old_difference:+.2f} đpt ở 5% '
          f'và {new_difference:+.2f} đpt ở 10%. Thứ hạng cần được kiểm tra theo '
          'ngân sách nhãn và qua nhiều seed; không mặc định mở ít hoặc nhiều block '
          'luôn tốt hơn.')
    prose(f'Full với 5% nhãn đạt Accuracy {percent(methods["5"]["finetune"]["accuracy"])}, '
          f'còn scratch với 10% nhãn đạt {percent(methods["10"]["scratch"]["accuracy"])}. '
          'Đây là một đối chiếu trực tiếp về hiệu quả nhãn trong lượt đo này; nhánh '
          'SSL vẫn có thêm dữ liệu không nhãn và chi phí pretraining như đã nêu.')
    for stage in ('linear', 'finetune', 'partial'):
        old_gap = (methods['5'][stage]['accuracy'] - methods['5']['scratch']['accuracy']) * 100
        new_gap = (methods['10'][stage]['accuracy'] - methods['10']['scratch']['accuracy']) * 100
        prose(f'Khoảng cách Accuracy {SHORT[stage]} so với scratch là '
              f'{old_gap:+.2f} đpt ở 5% và {new_gap:+.2f} đpt ở 10%. '
              'Đây là hai điểm của một seed; chưa thể suy ra xu hướng ở 1%, 25% hoặc 100%.')
    prose('Câu hỏi chính là: khi tăng gấp đôi nhãn dưới cùng cấu hình 100 epoch, '
          'nhánh nào cải thiện bao nhiêu và các kiểu lỗi cũ có giảm không? Kết quả '
          'chưa tách được ảnh hưởng của số bước tối ưu, độ nhạy với seed hoặc tương '
          'quan giữa các crop cùng scan. Không lấy hình vài ca để khẳng định texture, '
          'crop hay sai nhãn là nguyên nhân. Test đã được xem từ các vòng trước, nên '
          'phần phân tích là khám phá, không phải kiểm định độc lập sau nhiều lần tune.')
    prose('Thí nghiệm tiếp có cơ sở hơn sẽ ưu tiên lặp seed để kiểm tra độ ổn định '
          'hoặc cố định số updates để kiểm tra giả thuyết về thêm dữ liệu. Nếu nhắm '
          'một nhóm lỗi cụ thể, xác nhận nó trên validation và kiểm tra mẫu train '
          'trước khi chọn oversampling/augmentation. Lượt này chỉ đổi ngân sách nhãn, '
          'giữ cấu hình mô hình và không oversample.')
    end()

    section('references', 'Nguồn số liệu và cấu hình thực tế')
    for budget in ('5', '10'):
        links = []
        for stage in STAGES:
            url = methods[budget][stage]['wandb_url']
            if url:
                links.append(f'<a href="{html.escape(url)}">{SHORT[stage]}</a>')
                markdown.append(f'[{budget}% {SHORT[stage]}]({url})\n')
        paragraphs.append(f'<p>W&amp;B · {budget}%: ' + ' · '.join(links) + '</p>')
    paragraphs.append('<p><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/">'
                      'MedMNIST v2: dữ liệu và split theo CT scan</a> · '
                      '<a href="https://arxiv.org/abs/2111.06377">MAE paper</a></p>')
    markdown.extend(['[MedMNIST v2: dữ liệu và split theo CT scan]'
                     '(https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/)\n',
                     '[MAE paper](https://arxiv.org/abs/2111.06377)\n'])
    prose('label_comparison.json lưu metric tính lại từ dự đoán, index, lịch học, '
          'cấu hình thực và provenance. figures/ chứa PNG riêng. HTML nhúng hình và '
          'dữ liệu nên mở offline; các link W&B/nguồn chỉ truy cập mạng khi bấm.')
    paragraphs.append('<details><summary>Hash MAE và tập con</summary><pre>' + html.escape(
        json.dumps({'mae_sha256': data['source_checkpoint_sha256'],
                    'subsets': {key: budgets[key]['manifest'] for key in ('5', '10')}},
                   ensure_ascii=False, indent=2)) + '</pre></details>')
    end()
    names_json = json.dumps({stage: NAMES[stage] for stage in STAGES}, ensure_ascii=False)
    images_json = json.dumps(figure_payload, separators=(',', ':'))
    script = f'''const names={names_json};const images={images_json};
const stage=document.getElementById('matrix-stage'),budget=document.getElementById('matrix-budget');
Object.entries(names).forEach(([key,name])=>{{const o=document.createElement('option');
o.value=key;o.textContent=name;stage.append(o);}});stage.value='finetune';
function render(){{document.getElementById('matrix-image').src=images[stage.value+'_'+budget.value];
document.getElementById('matrix-caption').textContent=
names[stage.value]+' · '+budget.value+'% nhãn';}}
stage.addEventListener('change',render);budget.addEventListener('change',render);render();'''
    css = Path(__file__).with_name('followup.css').read_text('utf-8')
    document = ('<!doctype html><html lang="vi"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>CT-MAE · So sánh 5% và 10% nhãn</title><style>' + css + '</style><body>'
                '<main><header><p class="eyebrow">CT-MAE · LABEL BUDGET · SEED 42</p>'
                '<h1>Thêm nhãn có giúp mô hình tốt hơn?</h1>'
                '<p class="lead">So sánh 5% và 10% nhãn với cùng MAE checkpoint, '
                'kiến trúc và số epoch.</p><nav><a href="#results">Kết quả</a> · '
                '<a href="#classes">Từng organ</a> · <a href="#learning">Đường học</a> · '
                '<a href="#cases">Ảnh thay đổi</a></nav></header>'
                + ''.join(paragraphs) + '</main><script>' + script + '</script></body></html>')
    (output / 'label_efficiency_report.html').write_text(document, 'utf-8')
    (output / 'label_efficiency_report.md').write_text('\n'.join(markdown), 'utf-8')
