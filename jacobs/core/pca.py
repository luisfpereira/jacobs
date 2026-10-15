from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from polpo.ext.numpy.io import save_indexed_arrays
from polpo.ext.sklearn.decomposition import PCA, TruncatedPCA
from polpo.plot.pyplot import plot_subfigure_grid
from polpo.plot.seaborn import pairmatrix_legend_handles, plot_pairmatrix
from polpo.transform import CompositeTransform
from polpo.workflow.task import TaskRunner, task, task_family


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


class PCAPairplotsRunner(TaskRunner):
    """Generate one PDF with a PCA pairplot matrix per substructure."""

    def __init__(
        self,
        figures_dir,
        load_data,
        colors,
        n_components=5,
        prefix=None,
        state_dir=None,
    ):
        self.prefix = prefix
        super().__init__(state_dir or figures_dir / f".{self._name('pca_pairplots')}")

        self.figures_dir = figures_dir
        self.colors = colors
        self.n_components = n_components
        self._load_data = load_data

    def _name(self, name):
        return f"{self.prefix}_{name}" if self.prefix else name

    @classmethod
    def from_data(cls, figures_dir, data, colors, **kwargs):
        return cls(figures_dir, lambda: data, colors, **kwargs)

    @cached_property
    def data(self):
        with self.timer("load_data"):
            data = self._load_data()

        self.logger.info(
            "Loaded data in %.1fs",
            self.timer.duration("load_data"),
        )
        return data

    @task
    def pairplots(self):
        dfs = {
            struct: compute_pca_scores(deltas, self.n_components)
            for struct, deltas in self.data.items()
        }

        subjects = sorted({subject for df in dfs.values() for subject in df["subject"]})

        pc_cols = [f"PC{i + 1}" for i in range(self.n_components)]

        fig = plot_subfigure_grid(
            dfs,
            plot_pairmatrix,
            ncols=2,
            columns=pc_cols,
            groupby="subject",
            colors=self.colors,
        )

        handles = pairmatrix_legend_handles(subjects, self.colors)
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(subjects),
            frameon=False,
        )

        fig.savefig(self.figures_dir / self._name("pairplots"))
        plt.close(fig)


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
