from functools import cached_property

import numpy as np
from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.bcv.plot import plot_rank_errors_band
from polpo.neuroimaging.enigma.mesh import load_template
from polpo.plot.pyplot import plot_grid
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
from polpo.workflow.task import TaskRunner, task

from jacobs.core.lddmm.outputs import find_experiment_dirs


def load_rank_selection_inputs(struct, struct_dir):
    """Load template faces and flattened global deltas for one structure."""
    output = LddmmToGlobalOutput(struct_dir)
    deltas = output.view.global_deltas.map_values(
        lambda delta: np.asarray(delta).flatten()
    )
    return load_template(struct).faces, deltas


def select_rank(struct, struct_dir, selector, results_dir):
    """Fit a rank selector on one structure's global deltas and save the result."""
    faces, deltas = load_rank_selection_inputs(struct, struct_dir)
    selector.fit(faces, deltas).result_.to_dir(results_dir)


def load_rank_selection_results(outputs_dir):
    """Load rank-selection results per experiment, keyed by long structure name."""
    return {
        struct: GroupedMeshRankSelectionResult.from_dir(struct_dir / "rank_selection")
        for struct, struct_dir in find_experiment_dirs(
            outputs_dir, long_name=True
        ).items()
    }


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
