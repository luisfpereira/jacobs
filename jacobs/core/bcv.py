from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from polpo.bcv.mesh import GroupedMeshRankSelectionResult
from polpo.bcv.plot import plot_rank_errors_band
from polpo.latex import add_subfigures, compile_latex, make_figure_document
from polpo.plot.pyplot import plot_grid
from polpo.workflow.task import TaskRunner, task

from jacobs.core.pca import compute_attenuation


def get_rank_selection_dir(base_dir):
    return base_dir / "bcv" / "rank_selection"


def get_reconstruction_eval_dir(base_dir):
    return base_dir / "bcv" / "reconstruction_eval"


def get_longitudinal_fidelity_dir(base_dir):
    return base_dir / "bcv" / "longitudinal_fidelity"


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


def group_by_side(items):
    """Regroup {"Left X": value, "Right X": value} into {"X": {"Left": value, "Right": value}}."""
    pairs = {}
    for struct, value in items.items():
        side, name = struct.split(" ", 1)
        pairs.setdefault(name, {})[side] = value
    return pairs


class ReconstructionEvalFiguresRunner(TaskRunner):
    """Figures for reconstruction error and trend attenuation at the BCV rank."""

    def __init__(
        self, figures_dir, load_errors, load_atlas_rms, load_slopes, state_dir=None
    ):
        super().__init__(state_dir or figures_dir / ".reconstruction_eval")
        self.figures_dir = figures_dir
        self._load_errors = load_errors
        self._load_atlas_rms = load_atlas_rms
        self._load_slopes = load_slopes

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
    def attenuation(self):
        return {
            struct: (rank, compute_attenuation(input_slopes, reconstructed))
            for struct, (
                rank,
                input_slopes,
                reconstructed,
            ) in self._load_slopes().items()
        }

    def _plot_bilateral(self, data, plot_side, ylabel, name, reference=None):
        """Lay out left/right per structure; plot_side(ax, positions, values, color, side)."""
        colors = {"Left": "tab:blue", "Right": "tab:green"}
        offsets = {"Left": -0.2, "Right": 0.2}
        pairs = group_by_side(data)

        fig, ax = plt.subplots(figsize=(10, 4))

        for side in ("Left", "Right"):
            plot_side(
                ax,
                [i + offsets[side] for i in range(len(pairs))],
                [pair[side][1] for pair in pairs.values()],
                to_rgba(colors[side], alpha=0.5),
                side,
            )

        if reference is not None:
            ax.axhline(reference, color="gray", linestyle=":")

        labels = [
            f"{name_}\n(ranks {pair['Left'][0]} / {pair['Right'][0]})"
            for name_, pair in pairs.items()
        ]
        ax.set_xticks(range(len(pairs)), labels)
        ax.set_ylabel(ylabel)
        ax.legend()

        fig.tight_layout()
        fig.savefig(self.figures_dir / name)
        plt.close(fig)

    @staticmethod
    def _boxes(ax, positions, values, color, side):
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=0.35,
            patch_artist=True,
            medianprops={"color": "black"},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
        boxes["boxes"][0].set_label(side)

    @staticmethod
    def _bars(ax, positions, values, color, side):
        ax.bar(positions, values, width=0.4, color=color, edgecolor="black", label=side)

    @task
    def plot_errors(self):
        self._plot_bilateral(
            self.distances, self._boxes, "Reconstruction error (varifold)", "errors"
        )

    @task
    def plot_normalized_errors(self):
        self._plot_bilateral(
            self.normalized_distances,
            self._boxes,
            "Reconstruction error / atlas RMS distance",
            "normalized_errors",
        )

    @task
    def plot_attenuation(self):
        self._plot_bilateral(
            self.attenuation,
            self._bars,
            r"Attenuation factor",
            "attenuation",
            reference=1,
        )

    @task
    def merge_normalized_attenuation(self):
        paths = ["normalized_errors.pdf", "attenuation.pdf"]

        body = add_subfigures(paths, n_cols=1)
        text = make_figure_document(body, figures_path=self.figures_dir)

        compile_latex(text, self.figures_dir / "normalized_attenuation.pdf")
