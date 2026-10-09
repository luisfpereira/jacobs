import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.enigma.mesh import get_template_n_vertices
from polpo.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.jacobs.plot import update_mpl_params
from polpo.neuroi.naming import get_subcortical_struct_short_name
from polpo.plot.pyplot import plot_grid, plot_mean_errorbar
from polpo.workflow.task import TaskRunner, task


def plot_selected_ranks_errorbar(
    values,
    ranks,
    xlabel="Number of spatial partitions",
    set_xticks=True,
    ax=None,
):
    ax = plot_mean_errorbar(
        values,
        ranks,
        error="std",
        ax=ax,
    )

    if set_xticks:
        ax.set_xticks(values)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Selected rank")

    return ax


def load_diagnostic_ranks(struct_dir, diagnostics_folder="rank_selection_diagnostics"):
    diagnostics_dir = struct_dir / diagnostics_folder

    results = [
        GroupedMeshRankSelectionResult.from_dir(path)
        for path in diagnostics_dir.iterdir()
        if path.is_dir()
    ]

    ranks_by_n_parts = defaultdict(list)

    for result in results:
        ranks_by_n_parts[result.n_parts].append(result.rank)

    n_parts = sorted(ranks_by_n_parts)
    ranks = np.asarray([ranks_by_n_parts[n_part] for n_part in n_parts]).T

    return np.asarray(n_parts), ranks


class ExperimentFiguresRunner(TaskRunner):
    def __init__(self, figures_dir, results, state_dir=None):
        if state_dir is None:
            state_dir = figures_dir

        super().__init__(state_dir)
        self.figures_dir = figures_dir
        self.results = results

    @task
    def plot_ranks(self):
        fig, ax = plot_grid(
            self.results,
            plot_selected_ranks_errorbar,
            select=lambda key, item: item,
            figsize=(10, 3 * len(self.results) // 2),
        )

        fig.savefig(self.figures_dir / "rank_selection_n_parts")

    @task
    def plot_ranks_holdout_ratio(self):
        fig, axes = plot_grid(
            self.results,
            plot_selected_ranks_errorbar,
            select=lambda struct, item: (
                1 / item[0],
                item[1],
            ),
            figsize=(10, 3 * len(self.results) // 2),
            xlabel="Held-out vertices",
            set_xticks=False,
            sharex=True,
        )

        fig.savefig(self.figures_dir / "rank_selection_holdout_ratio")

    @task
    def plot_ranks_path_size(self):
        def select(struct, item):
            n_vertices = get_template_n_vertices(
                get_subcortical_struct_short_name(struct)
            )
            return n_vertices / item[0], item[1]

        fig, ax = plot_grid(
            results,
            plot_selected_ranks_errorbar,
            select=select,
            figsize=(10, 3 * len(self.results) // 2),
            xlabel="Vertices per spatial partition",
            set_xticks=False,
            sharex=False,
        )

        fig.savefig(self.figures_dir / "rank_selection_patch_size")


if __name__ == "__main__":
    outputs_dir = Path(os.environ["OUTPUTS_DIR"])
    figures_dir = Path(os.environ["FIGURES_DIR"]) / "rank_selection_diagnostics"

    update_mpl_params()

    struct_dirs = find_experiment_dirs(
        outputs_dir,
        long_name=True,
        interleave=True,
    )

    results = {
        struct: load_diagnostic_ranks(dir_) for struct, dir_ in struct_dirs.items()
    }

    figures_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentFiguresRunner(figures_dir, results)

    runner.run(overwrite=True)
