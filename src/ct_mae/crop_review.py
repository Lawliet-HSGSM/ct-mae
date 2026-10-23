"""Compare same-organ test crops and preserve their official source pixels."""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ct_mae.collect import LABELS_VI, png_data_url
from ct_mae.config import write_json
from ct_mae.download import file_md5, official_asset
from ct_mae.error_analysis import read_predictions
from ct_mae.followup_report import SHORT, STAGES, figure_save, plt

GROUPS = {
    'all_correct': 'Cả bốn model đúng',
    'scratch_to_full': 'Scratch sai → Full đúng',
    'full_to_partial': 'Full sai → Partial đúng',
}
DEFAULT_REFERENCES = [9916, 12619, 8329, 13003, 11590, 10180, 2495, 10267]


def choose_matches(images, labels, predictions, references, count=3):
    references = np.asarray(references, dtype=np.int64)
    if (not len(references) or len(np.unique(references)) != len(references)
            or references.min() < 0 or references.max() >= len(labels)):
        raise ValueError('References must be distinct official test indices')
    organ = int(labels[references[0]])
    if not np.all(labels[references] == organ):
        raise ValueError('Compare references from one organ at a time')
    if images.dtype != np.uint8 or images.shape != (len(labels), 128, 128):
        raise ValueError('Expected original uint8 test crops, 128 by 128')
    indices = np.flatnonzero(labels == organ)
    # Area-average only for retrieval; display and model pixels remain unchanged.
    features = images[indices].reshape(-1, 32, 4, 32, 4).mean(
        axis=(2, 4), dtype=np.float32,
    ).reshape(len(indices), -1) / 255
    correct = {stage: predictions[stage]['prediction'] == labels for stage in STAGES}
    pools = {
        'all_correct': np.logical_and.reduce(list(correct.values())),
        'scratch_to_full': ~correct['scratch'] & correct['finetune'],
        'full_to_partial': ~correct['finetune'] & correct['partial'],
    }
    result = []
    for index in references:
        distance = np.square(features - features[np.searchsorted(indices, index)]).mean(1)
        matches = {}
        for name, eligible in pools.items():
            positions = np.flatnonzero(eligible[indices] & (indices != index))
            order = np.argsort(distance[positions], kind='stable')[:count]
            matches[name] = [{'index': int(indices[position]), 'mse': float(distance[position])}
                             for position in positions[order]]
        result.append({'index': int(index), 'matches': matches})
    return {
        'organ': organ, 'labels_vi': LABELS_VI, 'groups': GROUPS, 'references': result,
        'test_organ_count': len(indices),
        'candidate_counts': {name: int(eligible[indices].sum())
                             for name, eligible in pools.items()},
        'matching': 'MSE of area-averaged 32x32 grayscale crops, fixed pixel scale /255; '
                    'same organ and required prediction outcomes; reference excluded; '
                    'ties resolved by official test index. This is visual-layout similarity, '
                    'not verified anatomical similarity. Controls may repeat across references.',
    }


def add_case_pixels(review, resolutions, labels, predictions):
    selected = set()
    for reference in review['references']:
        selected.add(reference['index'])
        selected.update(match['index'] for matches in reference['matches'].values()
                        for match in matches)
    cases = {}
    for index in sorted(selected):
        models = {}
        for stage, values in predictions.items():
            predicted = int(values['prediction'][index])
            models[stage] = {
                'prediction': predicted,
                'confidence': float(values['probability'][index, predicted]),
                'true_probability': float(values['probability'][index, labels[index]]),
            }
        cases[str(index)] = {
            'index': index, 'label': int(labels[index]), 'models': models,
            'images': {str(size): png_data_url(images[index])
                       for size, images in resolutions.items()},
        }
    review['cases'] = cases
    return review


def write_page(review, output):
    template = Path(__file__).with_suffix('.html').read_text('utf-8')
    style = Path(__file__).with_name('followup.css').read_text('utf-8')
    style += Path(__file__).with_suffix('.css').read_text('utf-8')
    script = Path(__file__).with_suffix('.js').read_text('utf-8')
    page = template.replace('__STYLE__', style).replace(
        '__DATA__', json.dumps(review, ensure_ascii=False, separators=(',', ':')),
    ).replace('__SCRIPT__', script)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'crop_review.json', review)
    (output / 'crop_review.html').write_text(page, 'utf-8')


def comparison_plot(review, images, output):
    references = review['references'][:2]
    fig, axes = plt.subplots(len(references), 4, figsize=(16, 4.8 * len(references)),
                             squeeze=False, layout='constrained')
    titles = ['Ảnh tham chiếu', 'Cả bốn đúng', 'Scratch sai → Full đúng',
              'Full sai → Partial đúng']
    for row, reference in enumerate(references):
        choices = [reference['index']] + [
            matches[0]['index'] if matches else None for matches in reference['matches'].values()
        ]
        for column, index in enumerate(choices):
            axis = axes[row, column]
            axis.set_axis_off()
            if index is None:
                continue
            case = review['cases'][str(index)]
            axis.imshow(images[index], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            lines = [titles[column] + f' · #{index}']
            for stage in ('scratch', 'finetune', 'partial'):
                value = case['models'][stage]
                lines.append(f'{SHORT[stage]}: {LABELS_VI[value["prediction"]]} '
                             f'({value["confidence"]:.1%})')
            axis.set_title('\n'.join(lines), fontsize=9)
    fig.suptitle(f'Cùng nhãn {LABELS_VI[review["organ"]]} · pixel gốc 128×128\n'
                 'Đối chiếu gần nhất theo bố cục pixel 32×32; ảnh đối chiếu có thể lặp lại',
                 fontsize=13)
    figure_save(fig, output, 'crop_review_pairs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--followup', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=DEFAULT_REFERENCES)
    args = parser.parse_args()
    images, sources, labels = {}, {}, None
    for size in (28, 64, 128):
        asset = official_asset(size)
        path = args.data_dir / asset['filename']
        digest = file_md5(path)
        if digest != asset['md5']:
            raise ValueError(f'Official data checksum mismatch: {path.name}')
        with np.load(path, allow_pickle=False) as saved:
            images[size] = saved['test_images']
            actual = saved['test_labels'].reshape(-1)
        if labels is not None and not np.array_equal(actual, labels):
            raise ValueError('Official test labels differ between resolutions')
        labels = actual
        sources[str(size)] = {'filename': path.name, 'md5': digest}
    original = json.loads(args.analysis.read_text('utf-8'))
    if not np.array_equal(labels, original['test_labels']):
        raise ValueError('Follow-up report test labels differ from official data')
    predictions = {}
    for stage in STAGES:
        root = args.followup if stage == 'partial' else args.baseline
        predictions[stage] = read_predictions(root / stage / 'test_predictions.npz', labels)
        if not np.array_equal(predictions[stage]['prediction'],
                              original['test_predictions'][stage]):
            raise ValueError('Prediction file differs from the original follow-up report')
    review = add_case_pixels(choose_matches(images[128], labels, predictions, args.indices),
                             images, labels, predictions)
    proof = []
    for index in args.indices:
        row = next((r for r in original['cases'] if r['index'] == index), None)
        if row is None:
            continue
        pixels = np.asarray(Image.open(BytesIO(base64.b64decode(row['image'].split(',')[1]))))
        if not np.array_equal(pixels, images[128][index]):
            raise ValueError('Original report PNG differs from source pixels')
        proof.append(index)
    verified = 0
    for case in review['cases'].values():
        for size, encoded in case['images'].items():
            pixels = np.asarray(Image.open(BytesIO(base64.b64decode(encoded.split(',')[1]))))
            if not np.array_equal(pixels, images[int(size)][case['index']]):
                raise ValueError('Review PNG changed source pixels')
            verified += 1
    review['provenance'] = {'sources': sources, 'original_pngs_equal': proof,
                            'roundtrip_pngs_verified': verified,
                            'budget': 'Original 5% labeled train, seed 42; no new training'}
    write_page(review, args.output)
    comparison_plot(review, images[128], args.output)
    print(json.dumps({'organ': LABELS_VI[review['organ']], 'references': args.indices,
                      'candidate_counts': review['candidate_counts'],
                      'pngs_verified': verified, 'report': str(args.output / 'crop_review.html')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
