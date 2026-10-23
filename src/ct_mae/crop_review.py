"""Compare same-organ test crops and preserve their official source pixels."""

import numpy as np

from ct_mae.collect import LABELS_VI, png_data_url
from ct_mae.followup_report import STAGES

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
