"""Vietnamese interpretation tied to recorded follow-up measurements."""

import html
import json
from pathlib import Path

import numpy as np

from ct_mae.followup_report import FEATURE_STAGES, NAMES, STAGES, embedded_figure, table


def pct(value):
    return f'{value * 100:.2f}%'


def blocks(data):
    metrics = data['metrics']
    full, partial, scratch = (metrics[key] for key in ('finetune', 'partial', 'scratch'))
    difference = (partial['accuracy'] - full['accuracy']) * 100
    yield ('Kết quả chính', [
        f'Partial fine-tuning đạt Accuracy {pct(partial["accuracy"])} và macro-F1 '
        f'{pct(partial["macro_f1"])} trên 17.778 ảnh test. So với full fine-tuning, '
        f'Accuracy thay đổi {difference:+.2f} điểm phần trăm. Đây là kết quả một seed, '
        'chưa chứng minh cách fine-tune nào luôn tốt hơn.',
        'Câu hỏi của lượt này: giữ nguyên phần lớn encoder sau MAE có giúp thích nghi với '
        'classification tốt hơn mở toàn bộ encoder không? Mở 3 block cuối là một lựa chọn '
        'thử nghiệm, chưa phải số block đã được tối ưu.',
    ])
    full_run, partial_run = data['runs']['finetune'], data['runs']['partial']
    yield ('Chi phí huấn luyện đã đo', [
        f'Partial cập nhật {partial_run["parameters"]["trainable"]:,} tham số so với '
        f'{full_run["parameters"]["trainable"]:,} của full, khoảng '
        f'{partial_run["parameters"]["trainable"] / full_run["parameters"]["trainable"]:.1%}. '
        f'Peak VRAM trong train giảm từ {full_run["peak_vram_bytes"] / 1024**3:.2f} '
        f'xuống {partial_run["peak_vram_bytes"] / 1024**3:.2f} GiB trên Tesla T4.',
        f'Tổng thời gian các vòng lặp train là {full_run["train_loop_seconds"] / 60:.1f} '
        f'phút ở full và {partial_run["train_loop_seconds"] / 60:.1f} phút ở partial. '
        f'Cả giai đoạn gồm validation/checkpoint/log tương ứng '
        f'{full_run["seconds_this_session"] / 60:.1f} và '
        f'{partial_run["seconds_this_session"] / 60:.1f} phút. Chi phí validation vẫn '
        'đáng kể vì cả hai dùng đầy đủ encoder. Đây là hai phiên Kaggle riêng, chưa '
        'phải benchmark tốc độ được lặp nhiều lần.',
    ])
    yield ('Thiết kế và kiểm tra tính công bằng', [
        'OrganAMNIST 128×128, grayscale, 11 lớp. Giữ split chính thức: train 34.561, '
        'validation 6.491, test 17.778. MAE học toàn bộ ảnh train và bỏ nhãn. Tất cả nhánh '
        'phân loại dùng cùng 1.728 ảnh có nhãn, chọn phân tầng seed 42; ngân sách 5% chỉ '
        'tính nhãn train, vẫn dùng toàn bộ nhãn validation để chọn checkpoint.',
        'Partial bắt đầu từ chính checkpoint MAE mà linear probing và full fine-tuning đã '
        'dùng. Chỉ mở block 10–12 (index 9–11), LayerNorm cuối và classifier. Patch embedding, '
        'CLS token và 9 block đầu giữ nguyên, ở chế độ eval. Các block được mở dùng '
        'stochastic depth 0.1 theo lịch của encoder; phần đóng băng không áp dụng nhiễu này.',
        'Số tham số train giảm chủ yếu tác động đến backpropagation, bộ nhớ optimizer '
        'và thời gian huấn luyện. Khi inference, cả full và partial vẫn chạy đủ 12 block '
        'với cùng kiến trúc encoder.',
        'Encoder: 12 block, chiều 384, 6 attention heads, patch 8×8. Decoder MAE: 2 block, '
        'chiều 128, 4 heads; che 75% patch, masked MSE trên pixel [0,1]. Khi phân loại, bỏ '
        'decoder, dùng ảnh đầy đủ → trung bình 256 patch token sau LayerNorm → Linear(384,11).',
        'Partial giữ cấu hình downstream của full: 100 epoch, AdamW, peak LR 3e-4, betas '
        '(0.9,0.999), weight decay 0.05; không decay bias/LayerNorm, epsilon 1e-8. Warmup '
        '5 epoch rồi cosine về 1e-6. Batch 32 × tích lũy 2 = 64; AMP FP16, seed 42. '
        'Augmentation train: affine ±5°, dịch tối đa 5%, bilinear, fill 0; không flip/crop. '
        'Validation/test chỉ float32 và chia 255.',
        'Mỗi epoch phân loại đi qua đúng 1.728 ảnh train; một epoch MAE đi qua 34.561 '
        'ảnh train không đọc nhãn. Validation không tạo bước cập nhật gradient.',
        'Chạy đủ 100 epoch, chọn validation macro-F1 cao nhất, hòa điểm giữ epoch sớm hơn; '
        'sau đó mới đánh giá checkpoint được chọn trên test. Không thay đổi LR hoặc số block '
        'sau khi nhìn kết quả test. So sánh này đo ảnh hưởng của cách mở encoder với một '
        'cấu hình LR cố định, chưa phải so sánh các phương pháp đã được tune tối ưu riêng.',
        'Giữ nguyên 9 block đầu không đủ chứng minh giữ được hành vi reconstruction. '
        'Lượt này chưa đo masked reconstruction sau adaptation; do đó không dùng kết '
        'quả classification để khẳng định partial bảo toàn objective MAE tốt hơn.',
        f'Kiểm tra trực tiếp checkpoint: {data["freeze_audit"]["frozen_tensors_unchanged"]} '
        f'tensor ở phần đóng băng giữ nguyên từng bit; '
        f'{data["freeze_audit"]["updated_tensors"]} tensor thuộc phần được mở đã thay đổi. '
        'SHA-256 checkpoint nguồn và hash tập con được lưu cùng kết quả.',
    ])
    yield ('Đọc kết quả theo từng cơ quan', [
        f'Accuracy tổng thể của scratch là {pct(scratch["accuracy"])}, full là '
        f'{pct(full["accuracy"])}. Cải thiện tổng thể không đồng nghĩa tất cả lớp đều '
        'cải thiện: cần xem precision, recall và F1 từng lớp.',
        'Precision của phổi trái trả lời: trong các ảnh model gọi là phổi trái, bao nhiêu '
        'ảnh thật sự là phổi trái? Recall trả lời: trong tất cả ảnh thật sự là phổi trái, '
        'model tìm đúng bao nhiêu? F1 kết hợp hai số này. Một lớp có recall cao vẫn có '
        'thể có F1 giảm nếu model gán quá nhiều ảnh của lớp khác vào nó.',
        'Confusion matrix: hàng là nhãn thật, cột là nhãn dự đoán. Đường chéo là đúng; '
        'ô ngoài đường chéo cho biết nhầm từ lớp nào sang lớp nào. Phần trăm được chuẩn '
        'hóa theo hàng để so các lớp có số ảnh khác nhau; số tuyệt đối giúp hiểu quy mô lỗi.',
    ])
    lung_full, lung_partial = (metrics[key]['per_class'][7] for key in ('finetune', 'partial'))
    yield ('Partial cải thiện ở đâu và đánh đổi gì?', [
        f'Precision phổi trái tăng từ {pct(lung_full["precision"])} lên '
        f'{pct(lung_partial["precision"])}; recall tương ứng '
        f'{pct(lung_full["recall"])} và {pct(lung_partial["recall"])}. Điều này phù '
        'hợp với việc giảm tim/lách bị gán nhầm thành phổi trái, thay vì cải thiện '
        'khả năng phân biệt hai lá phổi.',
        f'F1 thận trái ở full là {pct(full["per_class"][4]["f1"])} và ở partial là '
        f'{pct(partial["per_class"][4]["f1"])}; thận phải tương ứng '
        f'{pct(full["per_class"][5]["f1"])} và {pct(partial["per_class"][5]["f1"])}. '
        'Cả hai nhánh vẫn có lỗi đáng kể ở nhóm thận; Accuracy tổng thể cao hơn '
        'không giải quyết hết vấn đề này.',
    ])
    matrix = {key: np.array(metrics[key]['confusion']) for key in STAGES}
    for source, destination in ((7, 8), (8, 7), (3, 7), (10, 7)):
        values = ' → '.join(f'{NAMES[key]}: {matrix[key][source, destination]}'
                            for key in ('scratch', 'finetune', 'partial'))
        yield (f'{data["labels_vi"][source]} → {data["labels_vi"][destination]}', [values])
    yield ('Kiểm tra giả thuyết phổi trái / phải', [
        f'Full fine-tuning nhầm phổi trái → phải {matrix["finetune"][7, 8]} ảnh và '
        f'phải → trái {matrix["finetune"][8, 7]} ảnh. Partial tương ứng là '
        f'{matrix["partial"][7, 8]} và {matrix["partial"][8, 7]} ảnh.',
        'Giả thuyết của mentor là một điểm bắt đầu cần kiểm tra. Đếm nhầm trực tiếp giữa '
        'hai lá phổi ở bảng trên sẽ xác nhận hoặc bác bỏ nó đối với các checkpoint hiện '
        'tại. Không suy ra hai phổi bị nhầm nhau chỉ từ F1 phổi trái giảm. Đặc biệt cần '
        'xem ảnh tim/lách bị gán thành phổi trái và các ô nhầm khác.',
    ])
    outcomes = data['paired']['finetune_to_partial']
    yield ('Những ảnh thay đổi khi chuyển từ full sang partial', [
        f'Trên cùng ảnh test: partial sửa đúng {outcomes["recovered"]:,} ảnh mà full sai; '
        f'đồng thời làm sai {outcomes["regressed"]:,} ảnh mà full đúng. '
        f'{outcomes["both_wrong"]:,} ảnh bị cả hai dự đoán sai. Vì vậy phải xem cả ảnh '
        'được sửa lẫn ảnh bị giảm hiệu quả; chỉ chọn ảnh đẹp sẽ tạo kết luận thiên lệch.',
    ])
    yield ('Validation tốt có đồng nghĩa test tốt?', [
        f'Full có validation macro-F1 {pct(data["runs"]["finetune"]["best_score"])} '
        f'tại epoch được chọn, nhưng test macro-F1 là {pct(full["macro_f1"])}. '
        f'Partial tương ứng là {pct(data["runs"]["partial"]["best_score"])} và '
        f'{pct(partial["macro_f1"])}. Đây là chênh lệch giữa hai split, không phải '
        'độ suy giảm được đo trên cùng ảnh.',
        'Nguồn scan và mức độ khó có thể khác nhau giữa validation/test. Chỉ 16 scan '
        'validation nên cần thận trọng khi tune nhiều cấu hình trên cùng tập này. '
        'Chưa có scan ID để định lượng độ dao động theo ca; không suy ra số crop lớn '
        'đồng nghĩa đánh giá đã toàn diện.',
    ])
    yield ('Worst cases và giới hạn của việc nhìn ảnh', [
        'Worst case ở đây là ảnh cụ thể, không chỉ tên lớp có F1 thấp. Nhóm worst được '
        'xếp theo cross-entropy: −log xác suất của nhãn thật. Nó ưu tiên những ảnh model '
        'sai nhưng rất tự tin. Có thêm lỗi lấy ngẫu nhiên theo từng lớp và ảnh đúng để '
        'đối chiếu, vì nhóm cực đoan không đại diện toàn bộ tập test.',
        'Các gallery dùng pixel gốc 128×128, thang hiển thị 0–255 cố định, không điều '
        'chỉnh tương phản riêng từng ảnh. Xác suất hiển thị là đầu ra softmax chưa hiệu '
        'chuẩn, không phải độ chắc chắn y khoa. Quan sát ảnh có thể gợi ý độ tương phản, '
        'vị trí crop hoặc cấu trúc khó phân biệt; chưa đủ chứng minh nguyên nhân do texture.',
        'Texture là hoa văn cục bộ của cường độ pixel, ví dụ độ hạt hoặc những vùng sáng '
        'tối lặp lại. Oversampling tăng số lần một nhóm ảnh train được lấy, không làm ảnh '
        'nét hơn hay tạo texture mới. Chỉ nên thử nếu xác định được nhóm lỗi trên '
        'validation và có đủ mẫu train phù hợp; nhãn sai hoặc crop mơ hồ có thể bị học '
        'lặp lại nhiều hơn. Lượt này chưa oversample để giữ một thay đổi kiểm soát được.',
    ])
    yield ('Vì sao validation loss có thể thấp hơn train loss?', [
        'Ít ảnh validation không tự động làm loss trung bình thấp hơn. Nó làm ước lượng '
        'dao động hơn; ở đây các crop cùng CT scan còn có tương quan nên số crop không '
        'tương đương số ca độc lập. Split nguồn là 115/16/70 CT scan; file NPZ không có '
        'scan ID để phân tích hoặc bootstrap theo scan.',
        'Train loss trong lịch sử đo khi model đang cập nhật, có augmentation và '
        'stochastic depth; validation đo sau epoch ở eval trên ảnh sạch. Bảng chẩn đoán '
        'bên dưới đo lại cùng checkpoint ở eval trên train sạch, train augmentation cố '
        'định và validation sạch, giúp tách bớt khác biệt cách đo. Chênh lệch còn lại '
        'có thể liên quan độ khó, phân bố lớp hoặc scan, chưa xác định nguyên nhân duy nhất.',
    ])
    linear_loss = data['loss_diagnostics']['linear']['measurements']
    yield ('Đối chiếu trực tiếp ở linear probing', [
        f'Ở checkpoint được chọn, train sạch/eval có CE '
        f'{linear_loss["train_clean"]["loss"]:.4f}; train affine/eval '
        f'{linear_loss["train_augmented"]["loss"]:.4f}; validation sạch/eval '
        f'{linear_loss["val_clean"]["loss"]:.4f}. Affine làm CE train thay đổi '
        f'{linear_loss["train_augmented"]["loss"] - linear_loss["train_clean"]["loss"]:+.4f}. '
        'Đây là một lần augmentation cố định seed 42, epoch 0; chưa phải ước lượng '
        'trung bình qua nhiều phép biến đổi. Bảng này giúp kiểm tra giải thích về cách '
        'đo, không xác định được độ khó của từng CT scan.',
    ])
    yield ('PCA và SVD: nhìn biểu diễn encoder', [
        'Mỗi ảnh thành vector 384 chiều, đúng vector đi vào classifier. Trích ở eval, '
        'không augmentation. Có 5 mốc: ngẫu nhiên, sau MAE chưa học nhãn, supervised '
        'từ đầu, full và partial. Linear probing dùng encoder MAE đóng băng nên không '
        'cần một embedding plot trùng lặp.',
        'PCA trừ vector trung bình train rồi lấy hai hướng có phương sai lớn nhất. SVD '
        'ở đây không trừ trung bình: phân rã trực tiếp ma trận feature. PCA cũng có thể '
        'tính bằng SVD sau khi trừ mean; chúng không phải hai phương pháp hoàn toàn '
        'độc lập. Bản không trừ mean có thể dành một hướng để mô tả vector trung bình '
        'thay vì khác biệt giữa ảnh.',
        'Cả hai được fit trên 34.561 vector train không dùng nhãn, rồi biến đổi 17.778 '
        'vector test. Nhãn test chỉ tô màu và tính thống kê sau khi cố định mô hình. '
        'Mỗi encoder có hệ trục riêng, không so tọa độ tuyệt đối giữa các hình. Phần '
        'trăm phương sai cho biết 2D giữ lại bao nhiêu khác biệt của vector 384D; đám '
        'điểm chồng nhau trong 2D chưa chứng minh chúng không phân biệt được trong 384D.',
        'Bổ sung kiểm tra ở không gian gốc: lấy cố định tối đa 200 ảnh test mỗi lớp '
        '(2.200 ảnh tổng), tìm 10 láng giềng khác ảnh đó bằng cosine và đo tỷ lệ cùng '
        'nhãn. Đây là thống kê mô tả có dùng nhãn test, không phải classifier được train '
        'thêm. Crop cùng scan có thể gần nhau; chưa chứng minh encoder đã học giải '
        'phẫu chỉ từ một hình chiếu hoặc một điểm số.',
    ])
    yield ('Cách đọc insight và quyết định thí nghiệm tiếp', [
        'Định lượng trả lời lỗi xảy ra bao nhiêu, ở lớp nào, theo hướng nào. Định tính '
        'xem những ảnh cụ thể để gợi ý điều kiện xảy ra lỗi. Insight cần nối cả hai: '
        'một kiểu lỗi lặp lại, có thống kê hỗ trợ và có ví dụ đối chứng; một giả thuyết '
        'về nguyên nhân vẫn cần một thí nghiệm kiểm tra.',
        'Nếu một cặp organ nhầm nhiều trên validation, trước tiên xem crop và nhãn; '
        'sau đó mới cân nhắc lấy mẫu thêm đúng nhóm đó. Nếu partial giữ được chất lượng '
        'mà ít tham số/thời gian hơn, đó là lợi ích thực dụng; nếu giảm, mở ít block '
        'không phải lựa chọn tốt hơn với cấu hình hiện tại. Muốn so sánh chắc hơn, ưu tiên '
        'lặp 3 seed và chọn cấu hình bằng validation trước khi thử nhiều kiến trúc.',
        'Không dùng test để chọn số block, LR, oversampling hoặc mục tiêu học đối sánh '
        'rồi coi kết quả mới là kiểm định độc lập. Test đã được xem từ vòng trước: '
        'báo cáo này là phân tích khám phá. Một lượt partial không chứng minh quy tắc '
        '“chỉ full fine-tune khi domain shift lớn”; trong cùng domain vẫn có thể cần '
        'điều chỉnh biểu diễn từ reconstruction sang classification.',
    ])


def write_reports(data: dict, output: Path):
    metrics, runs = data['metrics'], data['runs']
    prose = list(blocks(data))
    observed_file = output / 'visual_observations.json'
    observations = json.loads(observed_file.read_text('utf-8')) if observed_file.exists() else []
    embedding_file = output / 'embedding_observations.json'
    embedding_notes = (json.loads(embedding_file.read_text('utf-8'))
                       if embedding_file.exists() else [])
    overview_rows = [
        [NAMES[stage], pct(metrics[stage]['accuracy']), pct(metrics[stage]['macro_f1']),
         metrics[stage]['errors'], runs[stage]['best_epoch'],
         f'{runs[stage]["parameters"]["trainable"]:,}',
         f'{runs[stage]["seconds_this_session"] / 60:.1f} phút'] for stage in STAGES
    ]
    summary_table = table(['Nhánh', 'Test Accuracy', 'Test macro-F1', 'Ảnh sai',
                           'Epoch chọn', 'Tham số train', 'Thời gian giai đoạn'], overview_rows)
    deltas = [(metrics['partial']['per_class'][i]['f1'] -
               metrics['finetune']['per_class'][i]['f1']) * 100 for i in range(11)]
    class_rows = [
        [data['labels_vi'][label], *[f'{metrics[stage]["per_class"][label]["f1"] * 100:.2f}'
                                    for stage in STAGES],
         f'{deltas[label]:+.2f}']
        for label in range(11)
    ]
    splits = ('train', 'labeled_train', 'val', 'test')
    count_rows = [[data['labels_vi'][label], *[data['counts'][split][label] for split in splits]]
                  for label in range(11)]
    loss_rows = []
    for stage in STAGES:
        measured = data['loss_diagnostics'][stage]['measurements']
        loss_rows.append([NAMES[stage], *[f'{measured[key]["loss"]:.4f}' for key in
                                        ('train_clean', 'train_augmented', 'val_clean')]])
    projection_rows = []
    for stage in FEATURE_STAGES:
        values = data['projections'][stage]
        projection_rows.append([
            NAMES[stage], pct(sum(values['pca']['centered_variance_ratio'])),
            pct(sum(values['svd']['energy_ratio'])),
            pct(sum(values['svd']['centered_variance_ratio'])),
            pct(values['neighbors']['agreement']),
        ])
    parts, markdown = [], ['# CT-MAE — Partial fine-tuning và phân tích lỗi',
                          'Báo cáo thực nghiệm khám phá · OrganAMNIST · seed 42\n']
    added = set()
    figures = {
        'Kết quả chính': ('metrics', 'F1 từng cơ quan và validation macro-F1 theo epoch.'),
        'Đọc kết quả theo từng cơ quan': ('confusion_matrices',
                                        'Confusion matrix: hàng thật, cột dự đoán.'),
        'Vì sao validation loss có thể thấp hơn train loss?': ('loss_curves',
                                                              'Loss trong quá trình training.'),
        'PCA và SVD: nhìn biểu diễn encoder': ('embedding_pca', 'PCA: màu theo organ.'),
    }
    for title, paragraphs in prose:
        parts.append(f'<section><h2>{html.escape(title)}</h2>')
        markdown.append(f'\n## {title}\n')
        for paragraph in paragraphs:
            parts.append(f'<p>{html.escape(paragraph)}</p>')
            markdown.append(paragraph + '\n')
        if title == 'Kết quả chính':
            parts.append(summary_table)
            markdown.append('| Nhánh | Accuracy | Macro-F1 | Sai | Epoch | Tham số train | Phút |\n'
                            '|---|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(
                                '| ' + ' | '.join(map(str, row)) + ' |' for row in overview_rows))
        if title == 'Thiết kế và kiểm tra tính công bằng':
            resource_rows = [[NAMES[stage], runs[stage]['environment'].get('gpu', 'CPU'),
                              runs[stage]['environment']['versions']['torch'],
                              f'{runs[stage]["peak_vram_bytes"] / 1024**3:.2f} GiB',
                              f'{runs[stage]["train_loop_seconds"] / 60:.1f} phút']
                             for stage in STAGES]
            parts.append(table(['Nhánh', 'GPU', 'PyTorch', 'Peak VRAM train',
                                'Chỉ vòng lặp train'], resource_rows))
            parts.append('<p>Thời gian giai đoạn gồm train, validation và ghi checkpoint/log; '
                         'không gồm tải dữ liệu và trích embedding. Cùng loại GPU vẫn có '
                         'thể khác tốc độ giữa các phiên Kaggle.</p>')
            parts.append(table(['Organ', 'Train', 'Train có nhãn', 'Validation', 'Test'],
                               count_rows))
            parts.append(embedded_figure(output, 'distributions',
                                          '5% train phân tầng; split chính thức theo CT scan.'))
            markdown.append('![Phân bố](figures/distributions.png)\n')
        if title == 'Đọc kết quả theo từng cơ quan':
            parts.append(table(['Organ', 'Scratch F1', 'Linear F1', 'Full F1', 'Partial F1',
                                'Partial − Full (đpt)'], class_rows))
            precision_rows = [
                [data['labels_vi'][label], metrics['partial']['per_class'][label]['support'],
                 *[f'{metrics[stage]["per_class"][label][measure] * 100:.2f}'
                   for stage in STAGES for measure in ('precision', 'recall')]]
                for label in range(11)
            ]
            parts.append('<details><summary>Precision / recall (%) và số ảnh test</summary>'
                         + table(['Organ', 'Support', 'Scratch P', 'Scratch R', 'Linear P',
                                  'Linear R', 'Full P', 'Full R', 'Partial P', 'Partial R'],
                                 precision_rows) + '</details>')
            matrices = {stage: np.array(metrics[stage]['confusion']) for stage in STAGES}
            off_diagonal = matrices['finetune'].copy()
            np.fill_diagonal(off_diagonal, 0)
            pairs = []
            for index in np.argsort(-off_diagonal.ravel(), kind='stable')[:12]:
                source, destination = divmod(int(index), 11)
                rate = off_diagonal[source, destination] / matrices['finetune'][source].sum()
                pairs.append([data['labels_vi'][source], data['labels_vi'][destination],
                              *[int(matrices[stage][source, destination])
                                for stage in ('scratch', 'finetune', 'partial')],
                              f'{rate:.1%}'])
            parts.append('<h3>12 hướng nhầm nhiều nhất của full fine-tuning</h3>' + table(
                ['Thật', 'Dự đoán', 'Scratch', 'Full', 'Partial', '% lớp thật ở Full'], pairs))
        if title == 'Worst cases và giới hạn của việc nhìn ảnh':
            if observations:
                parts.append('<aside><h3>Quan sát trực tiếp từ ảnh</h3>')
                for item in observations:
                    parts.append(f'<p>{html.escape(item)}</p>')
                    markdown.append(item + '\n')
                parts.append('</aside>')
            parts.append('''<div class="controls"><label>Nhóm ảnh
<select id="case-group"></select></label><label>Nhãn thật
<select id="case-label"><option value="all">Tất cả organ</option></select></label>
<label>Nhánh cần xem<select id="case-model"><option value="finetune">Full fine-tuning</option>
<option value="partial">Partial fine-tuning</option><option value="scratch">Scratch</option>
<option value="linear">Linear probing</option></select></label></div>
<p id="case-count" aria-live="polite"></p><div id="gallery" class="gallery"></div>''')
            for name, caption in (
                ('full_worst', '12 lỗi full có xác suất nhãn thật thấp nhất'),
                ('partial_worst', '12 lỗi partial có xác suất nhãn thật thấp nhất'),
                ('regression', 'Scratch đúng → full sai'),
                ('partial_recovers', 'Full sai → partial đúng'),
                ('partial_regresses', 'Full đúng → partial sai'),
                ('heart_to_lung', 'Tim bị full gán thành phổi trái'),
                ('spleen_to_lung', 'Lách bị full gán thành phổi trái'),
                ('kidney_to_spleen', 'Thận trái bị full gán thành lách'),
                ('correct_controls', 'Ảnh đúng để đối chiếu, chọn ngẫu nhiên theo lớp'),
                ('focus_controls', 'Ảnh đúng: tim, lách, phổi trái, thận trái, bàng quang'),
                ('heart_comparison', 'Đối chiếu: tim bị nhầm / tim đúng / phổi trái đúng'),
                ('kidney_comparison', 'Đối chiếu: thận trái bị nhầm / thận trái đúng / lách đúng'),
            ):
                parts.append(f'<details><summary>{caption}</summary>' + embedded_figure(
                    output, f'cases_{name}', caption) + '</details>')
                markdown.append(f'![{caption}](figures/cases_{name}.png)\n')
        if title == 'Vì sao validation loss có thể thấp hơn train loss?':
            parts.append(table(['Nhánh / best checkpoint', 'Train sạch · eval',
                                'Train affine · eval', 'Val sạch · eval'], loss_rows))
            markdown.append('| Nhánh | Train sạch | Train affine | Val sạch |\n'
                            '|---|---:|---:|---:|\n'
                            + '\n'.join('| ' + ' | '.join(row) + ' |' for row in loss_rows))
        if title == 'PCA và SVD: nhìn biểu diễn encoder':
            if embedding_notes:
                parts.append('<aside><h3>Quan sát từ các phép chiếu thực tế</h3>')
                for note in embedding_notes:
                    parts.append(f'<p>{html.escape(note)}</p>')
                    markdown.append(note + '\n')
                parts.append('</aside>')
            parts.append(table(['Encoder', 'PCA: phương sai 2D', 'SVD: năng lượng 2D',
                                'SVD: phương sai đã trừ mean', 'Cùng nhãn / 10 láng giềng'],
                               projection_rows))
            parts.append('''<div class="controls"><label>Encoder
<select id="embedding-stage"></select>
</label><label>Phép chiếu<select id="embedding-method"><option value="pca">PCA (trừ mean)</option>
<option value="svd">SVD (không trừ mean)</option></select></label><label>Hiển thị
<select id="embedding-filter"><option value="all">Tất cả ảnh test</option>
<option value="wrong">Ảnh sai ở nhánh được chọn</option></select></label></div>
<div id="embedding-legend" class="legend"></div>
<p id="embedding-info"></p><div class="canvas-wrap"><canvas id="embedding-canvas"
aria-label="Không gian embedding hai chiều"></canvas><div id="point-tooltip" hidden></div></div>
<p id="point-detail" aria-live="polite">Di chuột để xem index, nhãn thật và dự đoán.
Tắt/bật màu ở chú giải để kiểm tra từng cơ quan.</p>''')
            parts.append(embedded_figure(output, 'embedding_svd',
                                          'SVD không trừ mean: không đồng nhất với PCA.'))
            markdown.append('![SVD](figures/embedding_svd.png)\n')
            markdown.append('| Encoder | PCA variance | SVD energy | SVD centered variance | '
                            '10NN agreement |\n|---|---:|---:|---:|---:|\n' + '\n'.join(
                                '| ' + ' | '.join(row) + ' |' for row in projection_rows))
        if title in figures and title not in added:
            name, caption = figures[title]
            parts.append(embedded_figure(output, name, caption))
            markdown.append(f'![{caption}](figures/{name}.png)\n')
            added.add(title)
        parts.append('</section>')
    links = ''.join(f'<li><a href="{html.escape(runs[key]["wandb_url"])}">'
                    f'{html.escape(NAMES[key])} · W&amp;B</a></li>' for key in STAGES
                    if runs[key].get('wandb_url'))
    parts.append(f'<section><h2>Nguồn và khả năng tái lập</h2><ul>{links}</ul>')
    references = [
        ('MAE paper (He et al.)', 'https://arxiv.org/abs/2111.06377'),
        ('MedMNIST v2 — nguồn crop và split', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/'),
        ('MedMNIST / OrganAMNIST metadata',
         'https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py'),
        ('PCA — centering', 'https://scikit-learn.org/stable/modules/generated/'
         'sklearn.decomposition.PCA.html'),
        ('TruncatedSVD — uncentered', 'https://scikit-learn.org/stable/modules/generated/'
         'sklearn.decomposition.TruncatedSVD.html'),
    ]
    for title, url in references:
        parts.append(f'<p><a href="{url}">{title}</a></p>')
        markdown.append(f'[{title}]({url})\n')
    parts.append('<p>analysis.json lưu số đo và index; projections/*.npz lưu mean, basis và '
                 'tọa độ; figures/ chứa ảnh xuất riêng. Báo cáo HTML này mở offline, không '
                 'tải thư viện hoặc dữ liệu từ mạng. Các link nguồn chỉ mở khi bạn bấm.</p>')
    provenance = {key: data[key] for key in ('protocol', 'freeze_audit', 'embedding_provenance')}
    parts.append('<details><summary>Cấu hình và provenance đầy đủ</summary><pre>'
                 + html.escape(json.dumps(provenance, ensure_ascii=False, indent=2))
                 + '</pre></details></section>')
    root = Path(__file__).parent
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    document = '<!doctype html><html lang="vi"><meta charset="utf-8">' \
               '<meta name="viewport" content="width=device-width,initial-scale=1">' \
               '<title>CT-MAE · Partial fine-tuning và phân tích lỗi</title><style>' \
               + (root / 'followup.css').read_text('utf-8') + '</style><body><main>' \
               '<header><p class="eyebrow">CT-MAE · PHÂN TÍCH TIẾP THEO · SEED 42</p>' \
               '<h1>Partial fine-tuning<br>và mô hình sai ở đâu?</h1>' \
               '<p class="lead">Từ số liệu từng organ đến ảnh dự đoán sai và không gian '
    document += 'biểu diễn encoder. Toàn bộ kết quả lấy từ checkpoint đã chạy thực tế.</p>' \
                '</header>' + ''.join(parts) + '</main><script id="analysis-data" '
    document += 'type="application/json">' + payload + '</script><script>' \
                + (root / 'followup.js').read_text('utf-8') + '</script></body></html>'
    (output / 'followup_report.html').write_text(document, 'utf-8')
    (output / 'followup_report.md').write_text('\n'.join(markdown), 'utf-8')
