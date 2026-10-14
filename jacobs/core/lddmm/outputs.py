from polpo.dataset import MappedView
from polpo.neuroimaging.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)
from polpo.surface_mesh.transform import DeltaTransform
from polpo.workflow.lddmm_to_global.distances import (
    DistanceResults,
    MultiDistanceResults,
)
from polpo.workflow.lddmm_to_global.output import (
    LddmmToGlobalMultiOutput,
    LddmmToGlobalOutput,
)

from jacobs.core.paths import resolve_data_dir
from jacobs.core.study import get_session_to_week_map


def find_experiment_dirs(outputs_dir, long_name=False, interleave=True):
    """Find subcortical experiment directories.

    Parameters
    ----------
    outputs_dir : path-like
        Directory containing structure-specific experiment directories.
    long_name : bool
        Whether to use long structure names as dictionary keys.
    interleave : bool
        Whether to order left and right hemisphere structures in an
        interleaved sequence.

    Returns
    -------
    experiment_dirs : dict
        Mapping from structure names to experiment directories, ordered
        according to the requested structure ordering.
    """
    all_structs = get_all_subcortical_structs(interleave=interleave)
    order = {struct: i for i, struct in enumerate(all_structs)}

    dirs = sorted(
        (
            path
            for path in outputs_dir.iterdir()
            if path.is_dir() and path.name in order
        ),
        key=lambda path: order[path.name],
    )

    structs = (
        [get_subcortical_struct_long_name(dir_.name) for dir_ in dirs]
        if long_name
        else [dir_.name for dir_ in dirs]
    )

    return dict(zip(structs, dirs))


def load_struct_global_deltas(struct_dir):
    """Load global meshes as deltas from the global atlas."""
    output = LddmmToGlobalOutput(struct_dir)

    meshes = output.view.global_points.map_values(lambda point: point.as_surface())
    atlas = output.global_atlas_point.as_surface()

    return meshes.map_values(DeltaTransform(atlas))


def load_global_deltas(outputs_dir):
    """Load per-structure global deltas for all experiments in outputs_dir."""
    experiment_dirs = find_experiment_dirs(outputs_dir, long_name=True)

    return {
        struct: load_struct_global_deltas(struct_dir)
        for struct, struct_dir in experiment_dirs.items()
    }


def load_multi_outputs(outputs_dir, structs, data_dir=None):
    experiment_dirs = find_experiment_dirs(outputs_dir)
    key_to_week = get_session_to_week_map(resolve_data_dir(data_dir))

    outputs = {}

    for struct in structs:
        struct_long = get_subcortical_struct_long_name(struct)
        dirs = [experiment_dirs[f"{side}_{struct}"] for side in ("L", "R")]

        output = LddmmToGlobalMultiOutput([LddmmToGlobalOutput(path) for path in dirs])

        outputs[struct_long] = output.view.with_key_map(key_to_week)

    return outputs


def load_multi_distances(outputs, distances_folder):
    return {
        struct: MultiDistanceResults(
            {
                side: MappedView(
                    DistanceResults.from_dir(output_.path / distances_folder),
                    output.key_map.flatten(),
                )
                for side, output_ in zip(("L", "R"), output.outputs)
            }
        )
        for struct, output in outputs.items()
    }


def load_outputs(outputs_dir, data_dir=None):
    key_to_week = get_session_to_week_map(resolve_data_dir(data_dir))
    return {
        name: LddmmToGlobalOutput(path).view.with_key_map(key_to_week)
        for name, path in find_experiment_dirs(outputs_dir).items()
    }


def load_distances(outputs, distances_folder):
    return {
        name: MappedView(
            DistanceResults.from_dir(output.path / distances_folder),
            output.key_map.flatten(),
        )
        for name, output in outputs.items()
    }


def load_reconstruction_inputs(struct_dir):
    """Load global meshes, the atlas delta transform and metrics for one experiment."""
    output = LddmmToGlobalOutput(struct_dir)

    atlas = output.global_atlas.template.as_surface()
    meshes = output.view.global_points.map_values(lambda point: point.as_surface())

    metrics = {
        "squared_euclidean": output.instantiate_euclidean_metric().squared_dist,
        "squared_varifold": output.instantiate_varifold_metric().squared_dist,
    }

    return meshes.flatten(), DeltaTransform(atlas), metrics
