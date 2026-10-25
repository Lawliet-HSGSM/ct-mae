import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from ct_mae.adaptation_collect import audit_predictions
from ct_mae.adaptation_evaluate import calibration
from ct_mae.adaptation_report import galleries, positive_count_bound, table
from ct_mae.config import write_json
from ct_mae.error_analysis import classification_summary
from ct_mae.pipeline import sha256_file


def test_gallery_pairs_are_correct_and_source_pixels_are_lossless():
    labels = np.array([0, 1, 2, 3])
    images = np.arange(4 * 16 * 16).reshape(4, 16, 16).astype(np.uint8)

    def predictions(values):
        probabilities = np.full((4, 11), .001)
        probabilities[np.arange(4), values] = .99
        return {'prediction': np.array(values), 'probability': probabilities}

    source, updated = predictions([0, 0, 2, 0]), predictions([0, 1, 0, 0])
    data = {'selection': {s: {'variant': 'E'} for s in ('full', 'partial')}}
    values = {s: {'source': source, 'E': updated} for s in ('full', 'partial')}
    result = galleries(data, images, labels, values)
    groups = result['full']['E']
    for group, index in [('Được sửa', 1), ('Mới sai', 2), ('Cả hai sai', 3),
                         ('Cả hai đúng · ngẫu nhiên', 0)]:
        case = groups[group][0]
        assert case['index'] == index
        decoded = np.asarray(Image.open(BytesIO(base64.b64decode(case['image'].split(',')[1]))))
        np.testing.assert_array_equal(decoded, images[index])


def test_report_table_escapes_untrusted_text():
    rendered = table(['Column'], [['<script>unsafe()</script>']])
    assert '<script>' not in rendered
    assert '&lt;script&gt;' in rendered


def test_supcon_bound_counts_both_views_but_excludes_self():
    assert positive_count_bound([0, 1]) == 0.
    assert np.isclose(positive_count_bound([0, 0]), np.log(3))
    assert np.isclose(positive_count_bound([0, 0, 1]), 2 * np.log(3) / 3)


def test_collection_recomputes_calibration_and_binds_test_to_checkpoint(tmp_path):
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'example classifier weights')
    labels = np.array([0, 1])
    probability = np.full((2, 11), .01)
    probability[:, 0] = .9
    predicted = probability.argmax(1)
    np.savez_compressed(tmp_path / 'test_predictions.npz', index=np.arange(2), label=labels,
                        prediction=predicted, probability=probability)
    summary = classification_summary(labels, predicted)
    measured = {**summary, **calibration(labels, probability),
                'checkpoint_sha256': sha256_file(checkpoint)}
    write_json(tmp_path / 'test.json', measured)
    assert audit_predictions(tmp_path, labels, checkpoint)['accuracy'] == .5
    write_json(tmp_path / 'test.json', {**measured, 'nll': 0.})
    with pytest.raises(ValueError, match='nll differs'):
        audit_predictions(tmp_path, labels, checkpoint)
    write_json(tmp_path / 'test.json', measured)
    checkpoint.write_bytes(b'different classifier weights')
    with pytest.raises(ValueError, match='different checkpoint'):
        audit_predictions(tmp_path, labels, checkpoint)
