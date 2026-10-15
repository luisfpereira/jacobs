import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.datasets.jacobs.plot import update_mpl_params
from polpo.ext.numpy.io import load_indexed_array
from polpo.ext.sklearn.io import load_estimator
from polpo.io.json import load_json
from polpo.plot.pyplot import plot_grid
from polpo.workflow.lddmm_to_global.distances import DistanceResults
from polpo.workflow.task import TaskRunner, task

# TODO: plotting needs strong cleaning


def collect_results_per_struct(struct_dir, pca_folder="pca_reconstruction"):
    pca_dir = struct_dir / pca_folder
    ks = load_json(pca_dir / "manifest.json")["metadata"]["ks"]

    sdists = {
        name: load_indexed_array(pca_dir / f"squared_{name}.npz")[1]
        for name in ("varifold", "euclidean")
    }

    error_scale = {}
    template_mse = {}
    for name, folder in {
        "varifold": "post_dists/var",
        "euclidean": "post_dists/eucl",
    }.items():
        results = DistanceResults.from_dir(struct_dir / folder)
        error_scale[name] = (
            results.local_reconstruction_error().apply(lambda x: np.stack(x).mean())
            ** 2
        )
        template_mse[name] = results.global_atlas_distance().apply(
            lambda x: (np.stack(x) ** 2).mean(),
        )

    bcv_rank = GroupedMeshRankSelectionResult.from_dir(
        struct_dir / "rank_selection"
    ).rank

    pca = load_estimator(pca_dir / "pca.joblib")

    return {
        "ranks": ks,
        "sdists": sdists,
        "error_scale": error_scale,
        "template_mse": template_mse,
        "bcv_rank": bcv_rank,
        "pca": pca,
    }


if __name__ == "__main__":
    outputs_dir = Path(os.environ["OUTPUTS_DIR"])
    figures_dir = Path(os.environ["FIGURES_DIR"]) / "pca_reconstruction"

    update_mpl_params()

    struct_dirs = find_experiment_dirs(
        outputs_dir,
        long_name=True,
        interleave=True,
    )

    results = {
        struct: collect_results_per_struct(struct_dir)
        for struct, struct_dir in struct_dirs.items()
    }

    figures_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentPlotter(figures_dir, results)
    runner.run(overwrite=True)
