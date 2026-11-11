from functools import cached_property

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.bcv.plot import plot_rank_errors_band
from polpo.latex import add_subfigures, compile_latex, make_figure_document
from polpo.plot.pyplot import plot_grid
from polpo.workflow.task import TaskRunner, task

from jacobs.core.bilateral import (
    MARKER_STYLE,
    SIDE_COLORS,
    add_side_markers,
    group_by_side,
    plot_bars,
    plot_bilateral,
    plot_boxes,
)
from jacobs.core.paths import find_experiment_dirs
from jacobs.core.pca import compute_scaling_factor
from jacobs.core.style import (
    PAPER_DOCUMENT,
    PAPER_FIGSIZE,
    PAPER_RC,
    PAPER_SUBFIGURES,
)


def get_rank_selection_dir(base_dir):
    return base_dir / "bcv" / "rank_selection"


def get_reconstruction_eval_dir(base_dir):
    return base_dir / "bcv" / "reconstruction_eval"


def get_longitudinal_fidelity_dir(base_dir):
    return base_dir / "bcv" / "longitudinal_fidelity"


def get_rank_selection_figures_dir(figures_dir):
    return figures_dir / "bcv" / "rank_selection"


def get_evaluation_figures_dir(figures_dir):
    return figures_dir / "bcv" / "evaluation"


ERRORS_FIGURE = "errors"
SCALING_FIGURE = "scaling"
LEGEND_FIGURE = "legend"
NOISE_FLOOR_LABEL = "Noise floor"


def load_selected_rank(base_dir):
    """Load the BCV-selected rank for one experiment."""
    return GroupedMeshRankSelectionResult.from_dir(
        get_rank_selection_dir(base_dir)
    ).rank


def load_rank_selection_results(outputs_dir):
    """Load rank-selection results per experiment, keyed by long structure name."""
    return {
        struct: GroupedMeshRankSelectionResult.from_dir(
            get_rank_selection_dir(struct_dir)
        )
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
            shared_legend=None,
            sharex=True,
        )
        fig.savefig(self.figures_dir / "rank_errors")


class EvaluationFiguresRunner(TaskRunner):
    """Figures for reconstruction error and trend scaling factor at the BCV rank.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    load_errors : callable
        Loads squared reconstruction errors and ranks, per structure.
    load_slopes : callable
        Loads input and reconstructed slopes and ranks, per structure.
    load_noise_floors : callable or None
        Loads noise floors keyed by long structure name. If given, each error
        box is marked at its noise floor.
    rc : dict or None
        Matplotlib rc parameters applied while plotting and saving.
        If None, ``PAPER_RC``.
    figsize : tuple[float, float] or None
        Figure size in inches. If None, ``PAPER_FIGSIZE``.
    legend : bool
        Whether each figure has its own legend. Otherwise, the merged figure
        shows the single legend saved as ``legend``.
    state_dir : pathlib.Path or None
        Directory storing the task state.
    """

    def __init__(
        self,
        figures_dir,
        load_errors,
        load_slopes,
        load_noise_floors=None,
        rc=None,
        figsize=None,
        legend=False,
        state_dir=None,
    ):
        super().__init__(state_dir or figures_dir / ".evaluation")
        self.figures_dir = figures_dir
        self._load_errors = load_errors
        self._load_slopes = load_slopes
        self._load_noise_floors = load_noise_floors
        self.rc = PAPER_RC if rc is None else rc
        self.figsize = PAPER_FIGSIZE if figsize is None else figsize
        self.legend = legend

    @cached_property
    def distances(self):
        return {
            struct: (rank, np.sqrt(errors))
            for struct, (rank, errors) in self._load_errors().items()
        }

    @cached_property
    def scaling(self):
        return {
            struct: (rank, compute_scaling_factor(input_slopes, reconstructed))
            for struct, (
                rank,
                input_slopes,
                reconstructed,
            ) in self._load_slopes().items()
        }

    def _plot_bilateral(
        self, data, plot_side, ylabel, name, reference=None, markers=None
    ):
        """Plot (rank, values) per structure, with left / right ranks in labels.

        ``markers`` are optional noise floors, keyed by long structure name.
        """
        values = {struct: values for struct, (_, values) in data.items()}
        ranks = group_by_side({struct: rank for struct, (rank, _) in data.items()})
        tick_labels = {
            name_: f"{name_} ({pair['Left']} / {pair['Right']})"
            for name_, pair in ranks.items()
        }

        with mpl.rc_context(self.rc):
            fig, ax = plt.subplots(figsize=self.figsize, layout="constrained")
            plot_bilateral(ax, values, plot_side, tick_labels, reference)
            if markers is not None:
                add_side_markers(
                    ax, markers, list(tick_labels), label=NOISE_FLOOR_LABEL
                )

            ax.set_ylabel(ylabel)
            if self.legend:
                ax.legend()

            fig.savefig(self.figures_dir / name)
            plt.close(fig)

    @task
    def plot_errors(self):
        self._plot_bilateral(
            self.distances,
            plot_boxes,
            "Reconstruction error",
            ERRORS_FIGURE,
            markers=None
            if self._load_noise_floors is None
            else self._load_noise_floors(),
        )

    @task
    def plot_scaling(self):
        self._plot_bilateral(
            self.scaling,
            plot_bars,
            "Scaling factor",
            SCALING_FIGURE,
            reference=1,
        )

    @task
    def plot_legend(self):
        """Save a legend-only figure with the side colors (``legend``)."""
        handles = [
            Patch(facecolor=color, edgecolor="black", label=side)
            for side, color in SIDE_COLORS.items()
        ]
        if self._load_noise_floors is not None:
            handles.append(Line2D([], [], label=NOISE_FLOOR_LABEL, **MARKER_STYLE))

        with mpl.rc_context(self.rc):
            fig = plt.figure(figsize=(self.figsize[0], 0.25))
            fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
            fig.savefig(self.figures_dir / LEGEND_FIGURE)
            plt.close(fig)

    @task
    def merge_errors_scaling(self):
        body = add_subfigures(
            [f"{ERRORS_FIGURE}.pdf", f"{SCALING_FIGURE}.pdf"],
            width=PAPER_SUBFIGURES["width"],
            n_cols=1,
            footer=None if self.legend else _include_legend(self.figures_dir),
        )
        text = make_figure_document(
            body, figures_path=self.figures_dir, **PAPER_DOCUMENT
        )

        compile_latex(
            text, self.figures_dir / f"{ERRORS_FIGURE}-{SCALING_FIGURE}.pdf", crop=True
        )


def _include_legend(figures_dir):
    """LaTeX code including the legend-only figure at its natural size."""
    return rf"\includegraphics{{{(figures_dir / LEGEND_FIGURE).as_posix()}.pdf}}"


def make_errors_scaling_panel(
    figures_dirs,
    output_path,
    legend=True,
    subfigures=None,
    document=None,
):
    """Compile reconstruction errors over scaling factors, one column per pipeline.

    Parameters
    ----------
    figures_dirs : dict
        Reconstruction-eval figure directories, keyed by column title.
    output_path : pathlib.Path
        Path of the compiled PDF.
    legend : bool
        Whether to add the legend-only figure of the first pipeline once,
        below the grid.
    subfigures : dict or None
        Keyword arguments of ``add_subfigures``. If None, ``PAPER_SUBFIGURES``.
        The number of columns is the number of pipelines.
    document : dict or None
        Keyword arguments of ``make_figure_document``.
        If None, ``PAPER_DOCUMENT``.
    """
    subfigures = PAPER_SUBFIGURES if subfigures is None else subfigures
    document = PAPER_DOCUMENT if document is None else document

    names = (ERRORS_FIGURE, SCALING_FIGURE)
    paths = [
        (figures_dir / f"{name}.pdf").as_posix()
        for name in names
        for figures_dir in figures_dirs.values()
    ]

    body = add_subfigures(
        paths,
        titles=tuple(figures_dirs),
        footer=_include_legend(next(iter(figures_dirs.values()))) if legend else None,
        **(subfigures | {"n_cols": len(figures_dirs)}),
    )
    text = make_figure_document(body, figures_path=output_path.parent, **document)

    compile_latex(text, output_path, crop=True)
