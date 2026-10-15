from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from polpo.plot.pyplot import plot_grid, plot_subfigure_grid
from polpo.plot.seaborn import pairmatrix_legend_handles, plot_pairmatrix
from polpo.workflow.task import TaskRunner, task

from . import compute_pca_scores


class PCAPairplotsRunner(TaskRunner):
    """Generate one PDF with a PCA pairplot matrix per substructure."""

    def __init__(
        self,
        figures_dir,
        load_data,
        colors,
        n_components=5,
        prefix=None,
        state_dir=None,
    ):
        self.prefix = prefix
        super().__init__(state_dir or figures_dir / f".{self._name('pca_pairplots')}")

        self.figures_dir = figures_dir
        self.colors = colors
        self.n_components = n_components
        self._load_data = load_data

    def _name(self, name):
        return f"{self.prefix}_{name}" if self.prefix else name

    @classmethod
    def from_data(cls, figures_dir, data, colors, **kwargs):
        return cls(figures_dir, lambda: data, colors, **kwargs)

    @cached_property
    def data(self):
        with self.timer("load_data"):
            data = self._load_data()

        self.logger.info(
            "Loaded data in %.1fs",
            self.timer.duration("load_data"),
        )
        return data

    @task
    def pairplots(self):
        dfs = {
            struct: compute_pca_scores(deltas, self.n_components)
            for struct, deltas in self.data.items()
        }

        subjects = sorted({subject for df in dfs.values() for subject in df["subject"]})

        pc_cols = [f"PC{i + 1}" for i in range(self.n_components)]

        fig = plot_subfigure_grid(
            dfs,
            plot_pairmatrix,
            ncols=2,
            columns=pc_cols,
            groupby="subject",
            colors=self.colors,
        )

        handles = pairmatrix_legend_handles(subjects, self.colors)
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(subjects),
            frameon=False,
        )

        fig.savefig(self.figures_dir / self._name("pairplots"))
        plt.close(fig)


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


class PCAReconstructionFiguresRunner(TaskRunner):
    def __init__(self, figures_dir, load_summaries, state_dir=None):
        super().__init__(state_dir or figures_dir / ".pca_reconstruction")
        self.figures_dir = figures_dir
        self._load_summaries = load_summaries

    @cached_property
    def summaries(self):
        return self._load_summaries()

    @task
    def plot_mse(self):
        # TODO: need to handle better twin
        fig, ax = plot_grid(
            self.summaries,
            plot_reconstruction_mses,
            figsize=(10, 3 * len(self.summaries) // 2),
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
            self.summaries,
            plot_reconstruction_mses,
            figsize=(10, 3 * len(self.summaries) // 2),
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
        fig, _ = plot_grid(
            self.summaries,
            plot_pca_variance_with_rank,
            figsize=(10, 3 * len(self.summaries) // 2),
            share_legend=True,
            select=lambda _, summary: (
                summary["explained_variance_ratio"],
                summary["bcv_rank"],
            ),
        )
        fig.savefig(self.figures_dir / "explained_variance_ratio")
