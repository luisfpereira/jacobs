from functools import cached_property

from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.bcv.plot import plot_rank_errors_band
from polpo.plot.pyplot import plot_grid
from polpo.workflow.task import TaskRunner, task


def get_rank_selection_dir(base_dir):
    return base_dir / "bcv" / "rank_selection"


def get_reconstruction_eval_dir(base_dir):
    return base_dir / "bcv" / "reconstruction_eval"


def load_selected_rank(base_dir):
    """Load the BCV-selected rank for one experiment."""
    return GroupedMeshRankSelectionResult.from_dir(
        get_rank_selection_dir(base_dir)
    ).rank


def plot_rank_selection(result, ax=None):
    return plot_rank_errors_band(result.errors, best_rank=result.rank, ax=ax)


class RankSelectionFiguresRunner(TaskRunner):
    def __init__(self, figures_dir, load_results, state_dir=None):
        super().__init__(state_dir or figures_dir / ".rank_selection")
        self.figures_dir = figures_dir
        self._load_results = load_results

    @cached_property
    def results(self):
        return self._load_results()

    @task
    def plot_errors(self):
        fig, _ = plot_grid(
            self.results,
            plot_rank_selection,
            figsize=(10, 3 * len(self.results) // 2),
            share_legend=False,
        )
        fig.savefig(self.figures_dir / "rank_errors")
