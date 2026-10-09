import logging
import os
from pathlib import Path

import numpy as np

from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.ext.numpy.io import save_indexed_array
from polpo.ext.sklearn.decomposition import PCA, TruncatedPCA
from polpo.ext.sklearn.io import load_estimator, save_estimator
from polpo.surface_mesh.transform import DeltaTransform
from polpo.transform import CompositeTransform
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
from polpo.workflow.task import CompositeTaskRunner, TaskRunner, task, task_family


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
        Directory where fitted estimators and evaluation results are stored.
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

    @property
    def pca_path(self):
        return self.results_dir / "pca.joblib"

    @task
    def fit(self):
        X = self.transform(self.dataset.values_list())

        pca = PCA(
            n_components=int(np.max(self.ks)),
        ).fit(X)

        save_estimator(self.pca_path, pca)

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
        pca = load_estimator(self.pca_path)

        meshes = self.dataset.values_list()

        dissimilarities = []

        for k in self.ks:
            pca_k = TruncatedPCA.from_fitted(pca, n_components=k)

            transform = CompositeTransform([self.transform, pca_k])

            rec_meshes = transform.inverse(transform(meshes))

            dissimilarities.append(
                [
                    metric(mesh, rec_mesh)
                    for mesh, rec_mesh in zip(
                        meshes,
                        rec_meshes,
                    )
                ]
            )

        # (n_meshes, n_ks), so keys index axis 0
        dissimilarities = np.asarray(dissimilarities).T

        save_indexed_array(
            self.results_dir / f"{name}.npz",
            keys=self.dataset.keys_list(),
            data=dissimilarities,
        )


def prepare_inputs(struct_dir):
    output = LddmmToGlobalOutput(struct_dir)

    atlas = output.global_atlas.template.as_surface()
    dataset = output.view.global_points.map_values(lambda point: point.as_surface())

    metrics = {
        "squared_euclidean": output.instantiate_euclidean_metric().squared_dist,
        "squared_varifold": output.instantiate_varifold_metric().squared_dist,
    }

    return dataset.flatten(), DeltaTransform(atlas), metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    outputs_dir = Path(os.environ["OUTPUTS_DIR"])

    struct_dirs = find_experiment_dirs(
        outputs_dir,
        long_name=False,
    )

    ks = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 60, 80]

    runners = {
        struct: PCAReconstructionEvaluator(
            struct_dir / "pca_reconstruction_check",
            *prepare_inputs(struct_dir),
            ks=ks,
        )
        for struct, struct_dir in struct_dirs.items()
    }

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".pca_reconstruction_check",
    )

    runner.reset().run(
        overwrite=True,
        tasks=("L_Hipp",),
    )
