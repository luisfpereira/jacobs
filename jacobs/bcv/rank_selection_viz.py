import os
from pathlib import Path

from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.bcv.plot import plot_rank_errors_band
from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.datasets.jacobs.plot import update_mpl_params
from polpo.plot.pyplot import plot_grid
from polpo.workflow.task import TaskRunner, task


def plot_rank_selection(result, ax=None):
    # TODO: use reference length
    return plot_rank_errors_band(
        result.errors,
        best_rank=result.rank,
        ax=ax,
    )


class ExperimentFiguresRunner(TaskRunner):
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
    def plot_errors(self):
        fig, ax = plot_grid(
            self.results,
            plot_rank_selection,
            figsize=(10, 3 * len(self.results) // 2),
            share_legend=False,
        )

        fig.savefig(self.figures_dir / "rank_errors")


if __name__ == "__main__":
    outputs_dir = Path(os.environ["OUTPUTS_DIR"])
    figures_dir = Path(os.environ["FIGURES_DIR"]) / "rank_selection"

    update_mpl_params()

    struct_dirs = find_experiment_dirs(
        outputs_dir,
        long_name=True,
        interleave=True,
    )

    results = {
        struct: GroupedMeshRankSelectionResult.from_dir(dir_ / "rank_selection")
        for struct, dir_ in struct_dirs.items()
    }

    figures_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentFiguresRunner(figures_dir, results)

    runner.run(
        overwrite=True,
    )
