import base64
from io import BytesIO

import numpy as np
from PIL import Image

from ct_mae.crop_review import add_case_pixels, choose_matches


def review_fixture():
    labels = np.array([5, 5, 5, 5, 5, 4])
    images = np.broadcast_to(np.array([12, 10, 20, 30, 11, 12], dtype=np.uint8)[:, None, None],
                             (6, 128, 128)).copy()
    predictions = {}
    for stage in ('scratch', 'linear', 'finetune', 'partial'):
        predicted = labels.copy()
        predicted[0] = 0
        if stage == 'scratch':
            predicted[2] = 1
        if stage == 'finetune':
            predicted[3] = 1
        probability = np.full((6, 11), .01, dtype=np.float32)
        probability[np.arange(6), predicted] = .9
        predictions[stage] = {'prediction': predicted, 'probability': probability}
    return images, labels, predictions


def test_same_organ_controls_match_their_outcomes_and_layout():
    images, labels, predictions = review_fixture()
    result = choose_matches(images, labels, predictions, [0])
    matches = result['references'][0]['matches']
    # Index 5 has identical pixels but a different true organ, so it is ineligible.
    assert [row['index'] for row in matches['all_correct']] == [4, 1]
    assert [row['index'] for row in matches['scratch_to_full']] == [2]
    assert [row['index'] for row in matches['full_to_partial']] == [3]
    assert result['candidate_counts'] == {
        'all_correct': 2, 'scratch_to_full': 1, 'full_to_partial': 1,
    }
    result = choose_matches(images, labels, predictions, [1])
    assert 1 not in [r['index'] for r in result['references'][0]['matches']['all_correct']]


def test_review_pngs_roundtrip_original_pixels_at_every_resolution():
    images, labels, predictions = review_fixture()
    resolutions = {size: images[:, :size, :size].copy() for size in (28, 64, 128)}
    result = add_case_pixels(choose_matches(images, labels, predictions, [0]),
                             resolutions, labels, predictions)
    assert set(result['cases']) == {'0', '1', '2', '3', '4'}
    for index, case in result['cases'].items():
        for size, encoded in case['images'].items():
            decoded = np.asarray(Image.open(BytesIO(base64.b64decode(encoded.split(',')[1]))))
            np.testing.assert_array_equal(decoded, resolutions[int(size)][int(index)])
