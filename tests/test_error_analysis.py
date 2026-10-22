import numpy as np
import pytest

from ct_mae.error_analysis import (
    classification_summary,
    fit_projection,
    neighbor_agreement,
    paired_outcomes,
    read_predictions,
    select_errors,
)


def test_confusion_direction_precision_recall_and_paired_changes():
    labels = np.array([0, 0, 1, 1, 1])
    first = np.array([0, 1, 1, 0, 1])
    second = np.array([0, 0, 1, 0, 0])
    summary = classification_summary(labels, first, 2)
    assert summary['confusion'] == [[1, 1], [1, 2]]
    assert summary['per_class'][1]['precision'] == pytest.approx(2 / 3)
    assert summary['per_class'][0]['recall'] == 0.5
    assert paired_outcomes(labels, first, second) == {
        'both_correct': 2, 'recovered': 1, 'regressed': 1, 'both_wrong': 1,
    }


def test_predictions_are_reordered_by_index_and_reject_missing_rows(tmp_path):
    target = tmp_path / 'predictions.npz'
    labels = np.array([0, 1, 0])
    np.savez(target, index=[2, 0, 1], label=[0, 0, 1], prediction=[1, 0, 1],
             probability=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    values = read_predictions(target, labels, 2)
    np.testing.assert_array_equal(values['prediction'], [0, 1, 1])
    np.savez(target, index=[0, 0, 1], label=[0, 0, 1], prediction=[0, 0, 1],
             probability=[[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]])
    with pytest.raises(ValueError, match='indices'):
        read_predictions(target, labels, 2)


def test_projection_uses_only_train_and_matches_svd_subspace():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(60, 5)) * [5, 3, 2, 1, 0.1] + [100, 0, 0, 0, 0]
    test = rng.normal(size=(7, 5)) + 1000
    pca = fit_projection(train, test, centered=True)
    raw = fit_projection(train, test, centered=False)
    _, _, vectors = np.linalg.svd(train - train.mean(0), full_matrices=False)
    np.testing.assert_allclose(pca['components'].T @ pca['components'],
                               vectors[:2].T @ vectors[:2], atol=1e-10)
    np.testing.assert_allclose(pca['train_coordinates'].mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(pca['test_coordinates'],
                               (test - train.mean(0)) @ pca['components'].T)
    changed = fit_projection(train, test * 100, centered=True)
    np.testing.assert_allclose(changed['components'], pca['components'])
    assert raw['energy_ratio'].sum() > 0.99
    assert pca['centered_variance_ratio'].sum() < 1
    assert not np.allclose(raw['train_coordinates'].mean(0), 0)


def test_worst_cases_are_actual_errors_ranked_by_true_label_loss():
    labels = np.array([0, 1, 0, 1])
    probability = np.array([[0.9, 0.1], [0.99, 0.01], [0.2, 0.8], [0.1, 0.9]])
    selected = select_errors(labels, probability, count=2)
    np.testing.assert_array_equal(selected, [1, 2])


def test_neighbor_diagnostic_excludes_self_and_does_not_train_a_classifier():
    features = np.array([[1., 0.], [1., .01], [0., 1.], [.01, 1.]])
    labels = np.array([0, 1, 0, 1])
    values = neighbor_agreement(features, labels, np.arange(4), neighbors=1)
    assert values['agreement'] == 0
    assert values['pool_samples'] == 4
