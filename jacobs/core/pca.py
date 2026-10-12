from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from polpo.ext.sklearn.decomposition import PCA
from polpo.plot.pyplot import plot_subfigure_grid
from polpo.plot.seaborn import pairmatrix_legend_handles, plot_pairmatrix
from polpo.workflow.task import TaskRunner, task


def compute_pca_scores(deltas, n_components):
    """Compute PCA scores from subject/session deltas."""
    flattened = deltas.flatten()
    keys = flattened.keys_list()
    X = np.stack(flattened.values_list())

    if n_components > min(X.shape):
        raise ValueError(
            f"Cannot fit {n_components} PCs from deltas with shape {X.shape}."
        )

    pca = PCA(n_components=n_components).fit(X)
    scores = pca.transform(X)

    return pd.DataFrame(
        [
            {
                "subject": str(subject),
                "session": session,
                **{f"PC{i + 1}": float(score) for i, score in enumerate(z)},
            }
            for (subject, session), z in zip(keys, scores)
        ]
    )


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
