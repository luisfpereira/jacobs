"""Plot PCA pairplot matrices for global meshes across subcortical structures."""

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.datasets.jacobs.plot import get_subject_colors, update_mpl_params
from polpo.ext.sklearn.decomposition import PCA
from polpo.surface_mesh.transform import DeltaTransform
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
from polpo.workflow.task import TaskRunner, task


def compute_pca_scores(struct_dir, n_components):
    """Compute subject/session PCA scores of global template-relative deltas."""
    source = LddmmToGlobalOutput(struct_dir)

    meshes = source.view.global_points.map_values(lambda point: point.as_surface())
    atlas = source.global_atlas_point.as_surface()
    delta_transform = DeltaTransform(atlas)
    deltas = meshes.map_values(delta_transform)

    flattened = deltas.flatten()
    keys = flattened.keys_list()
    X = np.stack(flattened.values_list())

    if n_components > min(X.shape):
        raise ValueError(
            f"Cannot fit {n_components} PCs from deltas with shape {X.shape} "
            f"for {struct_dir}."
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


def plot_pairmatrix(subfig, df, pc_cols, subject_colors, title, alpha=0.6):
    """Draw a corner-style pairplot on a Matplotlib SubFigure."""
    n = len(pc_cols)
    axes = subfig.subplots(n, n, squeeze=False)
    subfig.suptitle(title)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                sns.kdeplot(
                    data=df,
                    x=pc_cols[i],
                    hue="subject",
                    palette=subject_colors,
                    fill=True,
                    alpha=0.2,
                    common_norm=False,
                    warn_singular=False,
                    legend=False,
                    ax=ax,
                )
                ax.set_yticks([])  # Diagonal density scale is not a PC scale.
            else:
                sns.scatterplot(
                    data=df,
                    x=pc_cols[j],
                    y=pc_cols[i],
                    hue="subject",
                    palette=subject_colors,
                    s=16,
                    alpha=alpha,
                    legend=False,
                    ax=ax,
                )

            ax.set_xlabel(pc_cols[j] if i == n - 1 else "")
            ax.set_ylabel(pc_cols[i] if j == 0 and i != j else "")
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            if j > 0 or i == j:
                ax.tick_params(labelleft=False)

    return axes


class PCAPairplotsRunner(TaskRunner):
    """Generate one PDF with a PCA pairplot matrix per substructure."""

    def __init__(
        self,
        outputs_dir,
        figures_dir,
        n_components=5,
        ncols=2,
        subject_colors=None,
        state_dir=None,
    ):
        figures_dir = Path(figures_dir)
        super().__init__(state_dir or figures_dir / ".pca_pairplots")

        self.outputs_dir = Path(outputs_dir)

        self.figures_dir = figures_dir
        self.n_components = n_components
        self.ncols = ncols
        self.subject_colors = subject_colors

    @task
    def pairplots(self):
        experiment_dirs = find_experiment_dirs(self.outputs_dir, long_name=True)

        if not experiment_dirs:
            raise ValueError(f"No structure directories found in {self.outputs_dir}")

        dfs = {
            struct: compute_pca_scores(path, self.n_components)
            for struct, path in experiment_dirs.items()
        }

        subjects = sorted({subject for df in dfs.values() for subject in df["subject"]})
        colors = self.subject_colors
        if colors is None:
            colors = get_subject_colors(subjects)

        nrows = math.ceil(len(dfs) / self.ncols)
        fig = plt.figure(
            figsize=(6.5 * self.ncols, 6.5 * nrows),
            layout="constrained",
        )
        subfigs = fig.subfigures(nrows, self.ncols, squeeze=False)
        pc_cols = [f"PC{i + 1}" for i in range(self.n_components)]

        for index, (struct, df) in enumerate(dfs.items()):
            subfig = subfigs[index // self.ncols, index % self.ncols]
            plot_pairmatrix(subfig, df, pc_cols, colors, title=struct)

        handles = [
            Line2D(
                [],
                [],
                color=colors[subject],
                marker="o",
                linestyle="none",
                label=subject,
            )
            for subject in subjects
        ]

        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(subjects),
            frameon=False,
        )

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(self.figures_dir / "pairplots")

        plt.close(fig)


if __name__ == "__main__":
    update_mpl_params()

    runner = PCAPairplotsRunner(
        outputs_dir=Path(os.environ["OUTPUTS_DIR"]),
        figures_dir=Path(os.environ["FIGURES_DIR"]) / "pca",
        n_components=5,
        ncols=2,
    )
    runner.run(overwrite=True)
