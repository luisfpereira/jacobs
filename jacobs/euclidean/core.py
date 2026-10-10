from polpo.datasets.jacobs.mesh import load_dataset as load_surface_dataset
from polpo.neuroimaging.enigma.mesh import read_geometry
from polpo.surface_mesh import Surface
from polpo.surface_mesh.registration import SurfaceRigidRegistration

from jacobs.core import (
    filter_pregnancy_weeks,
    get_session_to_week_map,
    resolve_data_dir,
    resolve_subject_ids,
    select_atlas_inputs,
)


def load_path_dataset(struct, data_dir=None, subject_ids=None):
    data_dir = resolve_data_dir(data_dir)
    subject_ids = resolve_subject_ids(subject_ids)

    derivative = "enigma"
    return load_surface_dataset(
        data_dir=data_dir,
        subject_subset=subject_ids,
        struct_subset=[struct],
        derivative=derivative,
        as_surface=False,
    )[struct]


def register_dataset(dataset):
    target = dataset.flatten().values_list()[0]

    register = SurfaceRigidRegistration(known_correspondences=True).against_same_target(
        target
    )

    return dataset.transform(register)


def prepare_inputs(struct, data_dir=None, subject_ids=None):
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
    paths = load_path_dataset(struct, data_dir, subject_ids)

    session_to_week = get_session_to_week_map()

    baseline_paths = select_atlas_inputs(paths, session_to_week)

    # retain subjects with suitable baseline inputs
    paths = paths.select_outer(baseline_paths.keys_list())

    observation_paths = filter_pregnancy_weeks(paths, session_to_week)

    # avoid loading shared observations twice
    selected_paths = observation_paths.merge(
        baseline_paths.drop_inner(observation_paths)
    )

    surfaces = selected_paths.map_values(lambda path: Surface(*read_geometry(path)))

    registered = register_dataset(surfaces)

    return (
        registered.select_inner(observation_paths),
        registered.select_inner(baseline_paths),
    )
