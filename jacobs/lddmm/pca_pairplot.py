"""Plot PCA pairplot matrices for global meshes across subcortical structures."""

from polpo.datasets.jacobs.plot import get_subject_colors
from polpo.surface_mesh.transform import DeltaTransform
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput

from jacobs.pca.core import PCAPairplotsRunner

from .core import find_experiment_dirs


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
