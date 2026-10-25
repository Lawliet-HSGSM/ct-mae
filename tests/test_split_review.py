import numpy as np

from ct_mae.split_review import accuracy_decomposition, conditional_errors, image_metrics, pick


def test_pixel_descriptors_have_known_values_and_respect_original_scale():
    step = np.array([[[0., 0., 1., 1.]] * 4], dtype=np.float32)
    measured = image_metrics(step)
    assert measured['mean'][0] == .5
    assert measured['std'][0] == .5
    assert measured['dark'][0] == .5
    assert np.isclose(measured['edge'][0], 1 / 6)
    integer = image_metrics((step * 255).astype(np.uint8))
    for key in measured:
        np.testing.assert_allclose(measured[key], integer[key])


def test_accuracy_decomposition_distinguishes_class_mix_from_same_class_errors():
    val_y, test_y = np.array([0] * 8 + [1] * 2), np.array([0] * 2 + [1] * 8)
    val_pred = np.array([0] * 8 + [1, 0])
    test_pred = np.array([0, 1] + [1] * 4 + [0] * 4)
    result = accuracy_decomposition(val_y, val_pred, test_y, test_pred)
    assert np.isclose(result['class_mix_effect'], -.3)
    assert np.isclose(result['within_class_effect'], -.1)
    assert np.isclose(result['test_mix_with_validation_recall'], .6)
    assert np.isclose(result['test']['accuracy'] - result['validation']['accuracy'],
                      result['class_mix_effect'] + result['within_class_effect'])


def test_gallery_sampling_is_reproducible_and_only_uses_requested_organ_and_group():
    labels = np.arange(33) % 3
    eligible = np.arange(33) % 2 == 0
    selected, count = pick(np.arange(33), labels, 1, seed=42, count=4, eligible=eligible)
    assert len(selected) == 4 and count == 5
    assert np.all(labels[selected] == 1) and eligible[selected].all()
    again, _ = pick(np.arange(33), labels, 1, seed=42, count=4, eligible=eligible)
    np.testing.assert_array_equal(selected, again)


def test_conditional_error_bins_use_train_cutoffs_and_keep_all_evaluation_rows():
    labels = np.repeat(np.arange(11), 4)
    x = np.tile([0., .2, .4, .8], 11)
    values = {k: x for k in ('mean', 'std', 'dark', 'edge')}
    predictions = {}
    for model in ('full', 'partial'):
        test = labels.copy()
        test[3] = 1
        predictions[model] = {'val': {'prediction': labels}, 'test': {'prediction': test}}
    measured = conditional_errors({**values, 'label': labels}, {'val': values, 'test': values},
                                   {'val': labels, 'test': labels}, predictions)
    groups = measured['0']['dark']
    assert sum(row['test']['samples'] for row in groups) == 4
    assert groups[3]['test']['models']['full'] == 1.
    assert groups[3]['val']['models']['full'] == 0.
