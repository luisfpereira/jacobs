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
def plot_reconstruction_mses(
    ks,
    sdists,
    error_scale,
    bcv_rank=None,
    template_mse=None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots()

    if template_mse is None:
        ax_r = ax.twinx()
    else:
        ax_r = ax

    for metric, color, ax_ in [
        ("varifold", "green", ax),
        ("euclidean", "blue", ax_r),
    ]:
        yvalues = sdists[metric].mean(axis=0)
        if template_mse is not None:
            yvalues /= template_mse[metric]

            label = "relative MSE"

        else:
            ax_.axhline(
                error_scale[metric],
                linestyle=":",
                color=color,
            )
            label = f"{metric.capitalize()} MSE"

        ax_.plot(
            ks,
            yvalues,
            label=metric,
            color=color,
        )

        ax_.set_ylabel(label)

    if bcv_rank is not None:
        ax_r.axvline(
            bcv_rank,
            label="BCV rank",
            color="red",
            linestyle="--",
            linewidth=0.8,
        )

    ax.set_xlabel("Rank")

    handles, labels = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(handles + handles_r, labels + labels_r)

    return (ax, ax_r)


def plot_pca_variance(
    explained_variance_ratio,
    levels=(0.9, 0.95),
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots()

    ranks = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative = np.cumsum(explained_variance_ratio)

    ax.plot(
        ranks,
        explained_variance_ratio,
        label="Explained variance",
    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Explained variance ratio")

    ax_r = ax.twinx()
    ax_r.plot(
        ranks,
        cumulative,
        linestyle="--",
        label="Cumulative explained variance",
    )
    ax_r.set_ylabel("Cumulative explained variance")
    ax_r.set_ylim(0, 1)

    for level in levels:
        ax_r.axhline(
            level,
            linestyle=":",
            alpha=0.5,
            label=f"{level:.0%}",
        )

    handles, labels = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(handles + handles_r, labels + labels_r)

    return ax, ax_r


def plot_pca_variance_with_rank(
    explained_variance_ratio,
    bcv_rank,
    levels=(0.9, 0.95),
    ax=None,
):
    ax, ax_r = plot_pca_variance(
        explained_variance_ratio,
        levels=levels,
        ax=ax,
    )

    ax.axvline(
        bcv_rank,
        linestyle="--",
        label="BCV rank",
        color="red",
    )

    ax.legend()

    return ax


class ExperimentPlotter(TaskRunner):
    def __init__(
        self,
        figures_dir,
        results,
        state_dir=None,
    ):
        if state_dir is None:
            state_dir = figures_dir

        super().__init__(state_dir)

        self.figures_dir = figures_dir
        self.results = results

    @task
    def plot_mse(self):
        # TODO: need to handle better twin
        fig, ax = plot_grid(
            self.results,
            plot_reconstruction_mses,
            figsize=(10, 3 * len(self.results) // 2),
            share_legend=True,
            select=lambda _, result: (
                result["ranks"],
                result["sdists"],
                result["error_scale"],
                result["bcv_rank"],
            ),
        )

        fig.savefig(self.figures_dir / "mses")

    @task
    def plot_normalized_mse(self):
        # TODO: need to handle better twin
        fig, ax = plot_grid(
            self.results,
            plot_reconstruction_mses,
            figsize=(10, 3 * len(self.results) // 2),
            share_legend=True,
            select=lambda _, result: (
                result["ranks"],
                result["sdists"],
                result["error_scale"],
                result["bcv_rank"],
                result["template_mse"],
            ),
        )

        fig.savefig(self.figures_dir / "normalized_mses")

    @task
    def plot_explained_variance(self):
        fig, ax = plot_grid(
            self.results,
            plot_pca_variance_with_rank,
            figsize=(10, 3 * len(self.results) // 2),
            share_legend=True,
            select=lambda _, result: (
                result["pca"].explained_variance_ratio_,
                result["bcv_rank"],
            ),
        )

        fig.savefig(self.figures_dir / "explained_variance_ratio")


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
