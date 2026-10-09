import os
from pathlib import Path

from polpo.dataset import MappedView, NestedKeyMap
from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.datasets.jacobs.plot import update_mpl_params
from polpo.datasets.jacobs.tabular import get_session_to_week
from polpo.distmat.plot import plot_distmat
from polpo.plot.pyplot import plot_grid
from polpo.workflow.lddmm_to_global.distances import DistanceResults
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
from polpo.workflow.task import CompositeTaskRunner, TaskRunner, task


class FiguresRunner(TaskRunner):
    """Base task runner for figure generation."""

    def __init__(self, figures_dir, distances, state_dir=None, prefix=""):
        if state_dir is None:
            state_dir = figures_dir

        super().__init__(state_dir)

        self.figures_dir = figures_dir
        self.distances = distances
        self.prefix = prefix

    def _get_figpath(self, name, prefixed=False):
        prefix = self.prefix if prefixed else ""
        return self.figures_dir / f"{prefix}{name}.pdf"

    def _plot_grid(self, name, data, plot, *, prefixed=False, **kwargs):
        fig, _ = plot_grid(data, plot, **kwargs)

        fig.tight_layout(
            rect=(0.04, 0.03, 0.96, 0.98),
        )

        fig.savefig(self._get_figpath(name, prefixed=prefixed))

    @task
    def global_dist_matrix(self):
        def _select(_, item):
            return (item.global_pairwise(),)

        self._plot_grid(
            "global-dist-matrix",
            self.distances,
            plot_distmat,
            prefixed=True,
            select=_select,
            show_ticks=False,
            figsize=(8, 18),
        )

    @task
    def local_dist_matrix(self):
        def _select(_, item):
            return (item.local_pairwise(),)

        self._plot_grid(
            "local-dist-matrix",
            self.distances,
            plot_distmat,
            prefixed=True,
            select=_select,
            show_ticks=False,
            figsize=(8, 18),
        )


def load_outputs(outputs_dir):
    experiment_dirs = find_experiment_dirs(outputs_dir)
    key_to_week = NestedKeyMap.from_inner_key_map(get_session_to_week())

    return {
        name: LddmmToGlobalOutput(path).view.with_key_map(key_to_week)
        for name, path in experiment_dirs.items()
    }


def load_distances(outputs, distances_folder):
    return {
        name: MappedView(
            DistanceResults.from_dir(output.path / distances_folder),
            output.key_map.flatten(),
        )
        for name, output in outputs.items()
    }


if __name__ == "__main__":
    outputs_dir = Path(os.environ["OUTPUTS_DIR"])
    figures_dir = Path(os.environ["FIGURES_DIR"]) / "lddmm"

    metrics = {
        "varifold": ("post_dists/var", "var"),
        "euclidean": ("post_dists/eucl", "eucl"),
        # "lddmm": ("post_dists/lddmm", "lddmm"),
    }

    outputs = load_outputs(outputs_dir)

    update_mpl_params()

    runners = {}
    for metric, (distances_folder, short) in metrics.items():
        distances = load_distances(outputs, distances_folder)

        runner = FiguresRunner(
            figures_dir,
            distances,
            prefix=f"{short}_",
            state_dir=figures_dir / f".{short}_distmats",
        )

        runners[metric] = runner

    CompositeTaskRunner(runners, figures_dir).run(
        overwrite=True,
        continue_on_error=False,
        tasks=(
            # "volumes",
            "euclidean",
            "varifold",
            # "lddmm",
            # "merge-volume",
            # "merge-metrics",
        ),
    )
