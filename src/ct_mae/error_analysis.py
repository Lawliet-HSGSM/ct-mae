"""Index-aligned error analysis and train-fitted linear projections; no model training."""

from pathlib import Path

import numpy as np


def read_predictions(path: Path, labels: np.ndarray, classes: int = 11) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in ('index', 'label', 'prediction', 'probability')}
    order = np.argsort(values['index'])
    values = {key: value[order] for key, value in values.items()}
    if not np.array_equal(values['index'], np.arange(len(labels))):
        raise ValueError(f'Prediction indices are missing, repeated or unexpected: {path}')
    if not np.array_equal(values['label'], labels):
        raise ValueError(f'Prediction labels disagree with official data: {path}')
    probability = values['probability']
    if probability.shape != (len(labels), classes) or not np.isfinite(probability).all():
        raise ValueError(f'Invalid class probabilities: {path}')
    if (probability < 0).any() or not np.allclose(probability.sum(1), 1, atol=2e-3):
        raise ValueError(f'Probabilities do not sum to one: {path}')
    if not np.array_equal(probability.argmax(1), values['prediction']):
        raise ValueError(f'Predictions disagree with class probabilities: {path}')
    return values


def classification_summary(labels: np.ndarray, predicted: np.ndarray, classes: int = 11):
    confusion = np.bincount(labels * classes + predicted, minlength=classes**2).reshape(
        classes, classes,
    )
    support, assigned = confusion.sum(1), confusion.sum(0)
    diagonal = confusion.diagonal()
    precision = np.divide(diagonal, assigned, out=np.zeros(classes), where=assigned > 0)
    recall = np.divide(diagonal, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * diagonal, support + assigned, out=np.zeros(classes),
                   where=support + assigned > 0)
    return {
        'accuracy': float(diagonal.sum() / len(labels)), 'macro_f1': float(f1.mean()),
        'errors': int(len(labels) - diagonal.sum()), 'confusion': confusion.tolist(),
        'per_class': [dict(label=i, support=int(support[i]), assigned=int(assigned[i]),
                           precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]))
                      for i in range(classes)],
    }


def paired_outcomes(labels, first, second):
    a, b = first == labels, second == labels
    return {'both_correct': int((a & b).sum()), 'recovered': int((~a & b).sum()),
            'regressed': int((a & ~b).sum()), 'both_wrong': int((~a & ~b).sum())}


def select_errors(labels, probability, count=12, eligible=None):
    wrong = probability.argmax(1) != labels
    if eligible is not None:
        wrong &= eligible
    candidates = np.flatnonzero(wrong)
    loss = -np.log(np.maximum(probability[candidates, labels[candidates]], 1e-30))
    return candidates[np.argsort(-loss, kind='stable')[:count]]


def fit_projection(train: np.ndarray, test: np.ndarray, *, centered: bool) -> dict:
    """Exact top right singular vectors via a small D×D Gram matrix (D=384).

    PCA = eigendecomposition of centered X.T@X. Raw SVD does not subtract the mean.
    Test values never enter the fit. Signs are fixed for stable artifacts.
    """
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError('Train and test must share the feature dimension')
    if not np.isfinite(train).all() or not np.isfinite(test).all() or len(train) < 2:
        raise ValueError('Projection requires finite features and at least two train rows')
    mean = train.mean(0) if centered else np.zeros(train.shape[1])
    x = train - mean
    gram = x.T @ x
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    components = vectors[:, order].T
    for component in components:
        if component[np.abs(component).argmax()] < 0:
            component *= -1
    coordinates = x @ components.T
    centered_energy = np.square(train - train.mean(0)).sum()
    return {
        'mean': mean, 'components': components,
        'train_coordinates': coordinates, 'test_coordinates': (test - mean) @ components.T,
        'singular_values': np.sqrt(np.maximum(eigenvalues[order], 0)),
        'energy_ratio': np.maximum(eigenvalues[order], 0) / max(float(gram.trace()), 1e-30),
        'centered_variance_ratio': np.square(coordinates - coordinates.mean(0)).sum(0)
        / max(float(centered_energy), 1e-30),
        'fit_samples': len(train), 'centered': centered,
    }


def balanced_pool(labels: np.ndarray, per_class=200, seed=42):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == label),
                   min(per_class, int((labels == label).sum())), replace=False)
        for label in np.unique(labels)
    ]))


def neighbor_agreement(features: np.ndarray, labels: np.ndarray, pool: np.ndarray,
                       neighbors: int = 10) -> dict:
    """Descriptive label agreement among other test examples; not a trained classifier."""
    x = features[pool].astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(neighbors, len(pool) - 1)
    nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    match = labels[pool][nearest] == labels[pool, None]
    return {'agreement': float(match.mean()), 'neighbors': k, 'pool_samples': len(pool),
            'distance': 'cosine on original pooled encoder features, no PCA/SVD',
            'per_class': [float(match[labels[pool] == i].mean())
                          for i in range(int(labels.max()) + 1)]}
