from functools import cached_property

import numpy as np
import pandas as pd
from polpo.ext.numpy.io import load_indexed_arrays, save_indexed_arrays
from polpo.ext.sklearn.decomposition import PCA, TruncatedPCA
from polpo.transform import CompositeTransform
from polpo.workflow.task import TaskRunner, task, task_family
from sklearn.linear_model import LinearRegression


def get_pca_reconstruction_eval_dir(base_dir):
    return base_dir / "pca" / "reconstruction_eval"


def compute_pca_scores(deltas, n_components):
    """Compute PCA scores from subject/session deltas."""
    flattened = deltas.flatten()
    keys = flattened.keys_list()
    X = np.stack(flattened.values_list())

    if n_components > min(X.shape):
        raise ValueError(
            f"Cannot fit {n_components} PCs from deltas with shape {X.shape}."
        )

    pca = PCA(n_components=n_components).fit(X)
    scores = pca.transform(X)

    return pd.DataFrame(
        [
            {
                "subject": str(subject),
                "session": session,
                **{f"PC{i + 1}": float(score) for i, score in enumerate(z)},
            }
            for (subject, session), z in zip(keys, scores)
        ]
    )


class PCAReconstructionEvaluator(TaskRunner):
    """Evaluate PCA mesh reconstruction across component counts.

    Fits PCA to a vector representation of a collection of meshes, reconstructs
    the meshes using different numbers of principal components, and evaluates
    the reconstructions with one or more metric callables.

    Each metric is exposed as an independent task and its reconstruction
    dissimilarities are persisted separately.

    Parameters
    ----------
    dataset : Dataset
        Dataset of meshes used to fit PCA and evaluate reconstruction. Dataset
        keys are used to index persisted reconstruction errors.
    transform : invertible transform
        Transform mapping meshes to the vector representation on which PCA is
        fitted.
    metrics : dict
        Mapping from task names to callables measuring dissimilarity between
        two meshes.
    ks : array-like
        Numbers of principal components used for reconstruction.
    results_dir : path-like
        Directory where evaluation results are stored.
    state_dir : path-like
        Directory where task execution state is stored. If ``None``,
        ``results_dir`` is used.
    """

    def __init__(
        self,
        results_dir,
        dataset,
        transform,
        metrics,
        ks,
        state_dir=None,
    ):
        if state_dir is None:
            state_dir = results_dir

        super().__init__(
            state_dir,
            metadata={
                "ks": ks,
                "metrics": list(metrics),
                "transform": transform.__class__.__name__,
            },
        )

        self.dataset = dataset
        self.transform = transform
        self.metrics = metrics

        self.ks = ks
        self.results_dir = results_dir

    @cached_property
    def pca(self):
        X = self.transform(self.dataset.values_list())

        pca = PCA(n_components=int(np.max(self.ks))).fit(X)
        return pca

    def _reconstruct(self, meshes, k):
        """Reconstruct meshes from the first k principal components."""
        pca_k = TruncatedPCA.from_fitted(self.pca, n_components=k)
        transform = CompositeTransform([self.transform, pca_k])

        return transform.inverse(transform(meshes))

    @task_family("metrics")
    def _run_metric(self, name, metric):
        """Evaluate and persist reconstruction dissimilarities for one metric.

        Parameters
        ----------
        name : str
            Name used to identify the metric task and its persisted results.
        metric : callable
            Function measuring dissimilarity between an original and a
            reconstructed mesh.
        """
        meshes = self.dataset.values_list()

        dissimilarities = {
            str(k): np.array(
                [
                    metric(mesh, rec)
                    for mesh, rec in zip(meshes, self._reconstruct(meshes, k))
                ]
            )
            for k in self.ks
        }

        save_indexed_arrays(
            self.results_dir / f"{name}.npz",
            keys=self.dataset.keys_list(),
            data=dissimilarities,
        )


def load_reconstruction_errors(results_dir, metric):
    """Load reconstruction errors as ks and an (n_meshes, n_ks) array."""
    _, data = load_indexed_arrays(results_dir / f"{metric}.npz")
    return [int(k) for k in data], np.stack(list(data.values()), axis=1)


def compute_log_volume_slope(times, meshes):
    """OLS slope of log-volume against time."""
    volumes = [mesh.to_polydata().volume for mesh in meshes]

    model = LinearRegression().fit(np.reshape(times, (-1, 1)), np.log(volumes))
    return model.coef_[0]


def compute_attenuation(input_slopes, reconstructed_slopes):
    """Regression through the origin of reconstructed on input slopes."""
    return input_slopes @ reconstructed_slopes / (input_slopes @ input_slopes)


class LongitudinalFidelityEvaluator(TaskRunner):
    """Log-volume slopes of input and PCA-reconstructed meshes, per subject and rank."""

    def __init__(self, results_dir, meshes, transform, ks, subjects, state_dir=None):
        ks = [int(k) for k in ks]
        subjects = list(subjects)

        super().__init__(
            state_dir or results_dir,
            metadata={
                "ks": ks,
                "subjects": subjects,
                "transform": type(transform).__name__,
            },
        )

        self.results_dir = results_dir
        self.meshes = meshes
        self.transform = transform
        self.ks = ks
        self.subjects = subjects

    @cached_property
    def pca(self):
        X = self.transform(self.meshes.flatten().values_list())
        return PCA(n_components=max(self.ks)).fit(X)

    def _reconstruct(self, meshes, k):
        pca_k = TruncatedPCA.from_fitted(self.pca, n_components=k)
        transform = CompositeTransform([self.transform, pca_k])
        return transform.inverse(transform(meshes))

    @task
    def slopes(self):
        slopes = {"input": []} | {str(k): [] for k in self.ks}

        for subject in self.subjects:
            sessions = self.meshes.get_outer(subject)
            times = np.asarray(sessions.keys_list(), dtype=float)
            meshes = sessions.values_list()

            slopes["input"].append(compute_log_volume_slope(times, meshes))
            for k in self.ks:
                slopes[str(k)].append(
                    compute_log_volume_slope(times, self._reconstruct(meshes, k))
                )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        save_indexed_arrays(
            self.results_dir / "slopes.npz",
            keys=self.subjects,
            data={name: np.asarray(values) for name, values in slopes.items()},
        )
