"""Plot PCA pairplot matrices for global meshes across subcortical structures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.datasets.jacobs.plot import get_subject_colors
from polpo.ext.sklearn.decomposition import PCA
from polpo.plot.pyplot import plot_subfigure_grid
from polpo.plot.seaborn import pairmatrix_legend_handles, plot_pairmatrix
from polpo.surface_mesh.transform import DeltaTransform
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
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
        deltas,
        figures_dir,
        colors,
        n_components=5,
        state_dir=None,
    ):
        super().__init__(state_dir or figures_dir / ".pca_pairplots")

        self.deltas = deltas
        self.figures_dir = figures_dir
        self.n_components = n_components

        self.colors = colors

    @task
    def pairplots(self):
        dfs = {
            struct: compute_pca_scores(deltas, self.n_components)
            for struct, deltas in self.deltas.items()
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

        fig.savefig(self.figures_dir / "pairplots")
        plt.close(fig)


def prepare_deltas(struct_dir):
    source = LddmmToGlobalOutput(struct_dir)

    meshes = source.view.global_points.map_values(lambda point: point.as_surface())
    atlas = source.global_atlas_point.as_surface()

    return meshes.map_values(DeltaTransform(atlas))


def make_runner(outputs_dir, figures_dir, n_components=5):
    experiment_dirs = find_experiment_dirs(outputs_dir, long_name=True)

    deltas = {struct: prepare_deltas(path) for struct, path in experiment_dirs.items()}

    return PCAPairplotsRunner(
        deltas=deltas,
        figures_dir=figures_dir,
        n_components=n_components,
        colors=get_subject_colors(),
    )
