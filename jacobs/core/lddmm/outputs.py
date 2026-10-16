import numpy as np
from polpo.dataset import MappedView
from polpo.neuroimaging.naming import (
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

from jacobs.core.paths import find_experiment_dirs, resolve_data_dir
from jacobs.core.study import get_session_to_week_map


def load_global_deltas(output):
    """Load global meshes as flattened deltas from the global atlas."""
    return load_global_meshes(output).map_values(
        DeltaTransform(load_global_atlas(output))
    )


def load_global_deltas_by_struct(outputs_dir):
    """Load global deltas for every experiment, keyed by long structure name."""
    return {
        struct: load_global_deltas(LddmmToGlobalOutput(struct_dir))
        for struct, struct_dir in find_experiment_dirs(
            outputs_dir, long_name=True
        ).items()
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


def load_global_meshes(output):
    """Load global meshes as surfaces, flattened over subject-session keys."""
    return output.view.global_points.map_values(lambda point: point.as_surface())


def load_global_atlas(output):
    """Load the global atlas as a surface."""
    return output.global_atlas.template.as_surface()


def load_atlas_rms_distance(struct_dir, metric="var"):
    """Load the RMS distance from global meshes to the global atlas."""
    results = DistanceResults.from_dir(struct_dir / "post_dists" / metric)
    return results.global_atlas_distance().apply(
        lambda x: np.sqrt((np.stack(x) ** 2).mean())
    )


def load_atlas_rms_distances(outputs_dir):
    """Load the RMS distance to the global atlas, keyed by long structure name."""
    return {
        struct: load_atlas_rms_distance(struct_dir)
        for struct, struct_dir in find_experiment_dirs(
            outputs_dir, long_name=True
        ).items()
    }
