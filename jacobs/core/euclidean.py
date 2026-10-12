import numpy as np
from polpo.dataset import NestedDataset
from polpo.datasets.jacobs.mesh import load_dataset as load_surface_dataset
from polpo.neuroimaging.enigma.mesh import read_geometry
from polpo.neuroimaging.naming import get_subcortical_struct_long_name
from polpo.surface_mesh import Surface
from polpo.surface_mesh.registration import SurfaceRigidRegistration
from polpo.surface_mesh.transform import DeltaTransform

from jacobs.core.paths import resolve_data_dir
from jacobs.core.study import (
    filter_pregnancy_weeks,
    get_session_to_week_map,
    resolve_subcortical_structs,
    resolve_subject_ids,
    select_observations_and_atlas_inputs,
)


def read_surface(path):
    return Surface(*read_geometry(path))


def load_path_dataset(struct, data_dir=None, subject_ids=None):
    data_dir = resolve_data_dir(data_dir)
    subject_ids = resolve_subject_ids(subject_ids)

    return load_surface_dataset(
        data_dir=data_dir,
        subject_subset=subject_ids,
        struct_subset=[struct],
        derivative="enigma",
        as_surface=False,
    )[struct]


def register_dataset(dataset):
    target = dataset.flatten().values_list()[0]

    register = SurfaceRigidRegistration(known_correspondences=True).against_same_target(
        target
    )

    return dataset.transform(register)


def load_registered_dataset(struct, data_dir=None, subject_ids=None):
    """Load and rigidly register pregnancy-window observations."""

    data_dir = resolve_data_dir(data_dir)
    paths = load_path_dataset(struct, data_dir, subject_ids)

    session_to_week = get_session_to_week_map(data_dir)
    paths = filter_pregnancy_weeks(paths, session_to_week)

    surfaces = paths.map_values(read_surface)
    return register_dataset(surfaces)


def load_registered_dataset_with_baselines(struct, data_dir=None, subject_ids=None):
    """Prepare rigidly aligned observation and baseline surface datasets.

    Load surfaces, select observations and baseline inputs, and rigidly
    register all meshes to a common reference.

    Parameters
    ----------
    struct : str
        Subcortical structure to load.
    data_dir : path-like
        Root directory containing the dataset.
    subject_ids : array-like
        Subject identifiers to include.

    Returns
    -------
    observations : NestedDataset
        Registered surfaces within the pregnancy window (weeks 0-42),
        retaining all observations from non-pregnancy subjects.
    baseline_inputs : NestedDataset
        Registered surfaces selected for baseline estimation.
    """
    data_dir = resolve_data_dir(data_dir)
    paths = load_path_dataset(struct, data_dir, subject_ids)

    session_to_week = get_session_to_week_map(data_dir)
    observation_paths, baseline_paths = select_observations_and_atlas_inputs(
        paths, session_to_week
    )

    # avoid loading shared observations twice
    selected_paths = observation_paths.merge(baseline_paths, strict=False)

    surfaces = selected_paths.map_values(read_surface)

    registered = register_dataset(surfaces)

    return (
        registered.select_inner(observation_paths.nested_keys()),
        registered.select_inner(baseline_paths.nested_keys()),
    )


def compute_baseline_deltas(observations, baseline_inputs):
    """Express observations as deltas from each subject's mean baseline."""
    deltas = {}

    for subject, sessions in observations.iter_outer():
        baselines = baseline_inputs.get_outer(subject).values_list()

        baseline = Surface(
            np.mean([mesh.vertices for mesh in baselines], axis=0),
            baselines[0].faces,
        )

        deltas[subject] = sessions.map_values(DeltaTransform(baseline)).data

    return NestedDataset(deltas)


def flatten_vertices(observations):
    """Flatten mesh vertices into feature vectors."""
    return observations.map_values(lambda mesh: mesh.vertices.reshape(-1))


def load_features(
    structs=None, data_dir=None, subject_ids=None, subtract_baseline=True
):
    """Load per-structure PCA features: baseline deltas or raw vertices."""
    structs = resolve_subcortical_structs(structs)

    def load_struct_features(struct):
        if subtract_baseline:
            return compute_baseline_deltas(
                *load_registered_dataset_with_baselines(struct, data_dir, subject_ids)
            )
        return flatten_vertices(load_registered_dataset(struct, data_dir, subject_ids))

    return {
        get_subcortical_struct_long_name(struct): load_struct_features(struct)
        for struct in structs
    }
