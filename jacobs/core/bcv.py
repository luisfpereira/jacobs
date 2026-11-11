from functools import cached_property

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.bcv.plot import plot_rank_errors_band
from polpo.latex import add_subfigures, compile_latex, make_figure_document
from polpo.plot.pyplot import plot_grid
from polpo.workflow.task import TaskRunner, task

from jacobs.core.pca import compute_scaling_factor
from jacobs.core.style import (
    COLORS,
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


SCALING_FIGURE = "scaling"
LEGEND_FIGURE = "legend"


def get_errors_figure_name(normalize):
    """Name of the reconstruction error figure."""
    return "normalized_errors" if normalize else "errors"


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
            shared_legend=None,
            sharex=True,
        )
        fig.savefig(self.figures_dir / "rank_errors")


SIDE_COLORS = {"Left": COLORS["left"], "Right": COLORS["right"]}


def group_by_side(items):
    """Regroup {"Left X": value, "Right X": value} into {"X": {"Left": value, "Right": value}}."""
    pairs = {}
    for struct, value in items.items():
        side, name = struct.split(" ", 1)
        pairs.setdefault(name, {})[side] = value
    return pairs


class ReconstructionEvalFiguresRunner(TaskRunner):
    """Figures for reconstruction error and trend scaling factor at the BCV rank.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    load_errors : callable
        Loads squared reconstruction errors and ranks, per structure.
    load_atlas_rms : callable
        Loads atlas RMS distances, per structure.
    load_slopes : callable
        Loads input and reconstructed slopes and ranks, per structure.
    normalize : bool
        Whether errors are plotted relative to the atlas RMS distance
        (``normalized_errors``) or raw (``errors``), and merged with the
        scaling factors into ``normalized_errors-scaling`` or ``errors-scaling``.
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
        load_atlas_rms,
        load_slopes,
        normalize=True,
        rc=None,
        figsize=None,
        legend=False,
        state_dir=None,
    ):
        super().__init__(state_dir or figures_dir / ".reconstruction_eval")
        self.figures_dir = figures_dir
        self._load_errors = load_errors
        self._load_atlas_rms = load_atlas_rms
        self._load_slopes = load_slopes
        self.normalize = normalize
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
    def normalized_distances(self):
        rms = self._load_atlas_rms()
        return {
            struct: (rank, distances / rms[struct])
            for struct, (rank, distances) in self.distances.items()
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

    def _plot_bilateral(self, data, plot_side, ylabel, name, reference=None):
        """Lay out left/right per structure; plot_side(ax, positions, values, color, side)."""
        offsets = {"Left": -0.2, "Right": 0.2}
        pairs = group_by_side(data)

        with mpl.rc_context(self.rc):
            fig, ax = plt.subplots(figsize=self.figsize, layout="constrained")

            for side in ("Left", "Right"):
                plot_side(
                    ax,
                    [i + offsets[side] for i in range(len(pairs))],
                    [pair[side][1] for pair in pairs.values()],
                    SIDE_COLORS[side],
                    side,
                )

            if reference is not None:
                ax.axhline(reference, color="gray", linestyle=":")

            # left / right ranks after each structure name
            labels = [
                f"{name_} ({pair['Left'][0]} / {pair['Right'][0]})"
                for name_, pair in pairs.items()
            ]
            ax.set_xticks(
                range(len(pairs)),
                labels,
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            ax.set_ylabel(ylabel)
            if self.legend:
                ax.legend()

            fig.savefig(self.figures_dir / name)
            plt.close(fig)

    @staticmethod
    def _boxes(ax, positions, values, color, side):
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=0.3,
            patch_artist=True,
            medianprops={"color": "black"},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
        boxes["boxes"][0].set_label(side)

    @staticmethod
    def _bars(ax, positions, values, color, side):
        ax.bar(positions, values, width=0.4, color=color, edgecolor="black", label=side)

    @property
    def _errors_name(self):
        return get_errors_figure_name(self.normalize)

    @task
    def plot_errors(self):
        if self.normalize:
            data = self.normalized_distances
            ylabel = "Reconstruction error /\natlas RMS distance"
        else:
            data = self.distances
            ylabel = "Reconstruction error\n(varifold)"

        self._plot_bilateral(data, self._boxes, ylabel, self._errors_name)

    @task
    def plot_scaling(self):
        self._plot_bilateral(
            self.scaling,
            self._bars,
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

        with mpl.rc_context(self.rc):
            fig = plt.figure(figsize=(self.figsize[0], 0.25))
            fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
            fig.savefig(self.figures_dir / LEGEND_FIGURE)
            plt.close(fig)

    @task
    def merge_errors_scaling(self):
        body = add_subfigures(
            [f"{self._errors_name}.pdf", f"{SCALING_FIGURE}.pdf"],
            width=PAPER_SUBFIGURES["width"],
            n_cols=1,
            footer=None if self.legend else _include_legend(self.figures_dir),
        )
        text = make_figure_document(
            body, figures_path=self.figures_dir, **PAPER_DOCUMENT
        )

        compile_latex(text, self.figures_dir / f"{self._errors_name}-scaling.pdf")


def _include_legend(figures_dir):
    """LaTeX code including the legend-only figure at its natural size."""
    return rf"\includegraphics{{{(figures_dir / LEGEND_FIGURE).as_posix()}.pdf}}"


def make_errors_scaling_panel(
    figures_dirs,
    output_path,
    normalize=True,
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
    normalize : bool
        Whether to use errors relative to the atlas RMS distance.
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

    names = (get_errors_figure_name(normalize), SCALING_FIGURE)
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

    compile_latex(text, output_path)
