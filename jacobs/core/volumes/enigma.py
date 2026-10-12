import pyvista as pv
from polpo.datasets.jacobs.mesh import load_dataset
from polpo.neuroimaging.enigma.mesh import read_geometry
from polpo.neuroimaging.naming import get_subcortical_struct_long_name

from jacobs.core.paths import resolve_data_dir
from jacobs.core.study import (
    filter_pregnancy_weeks,
    get_session_to_week_map,
    group_by_pregnancy,
    resolve_subcortical_structs,
    resolve_subject_ids,
)


def _compute_mesh_volume(path):
    return pv.PolyData.from_regular_faces(*read_geometry(path)).volume


def _load_struct_volumes(paths, session_to_week):
    volumes = filter_pregnancy_weeks(paths, session_to_week).map_values(
        _compute_mesh_volume
    )
    return group_by_pregnancy(volumes.map_keys(session_to_week))


def load_volumes(structs=None, subject_ids=None, data_dir=None):
    derivative = "enigma"
    data_dir = resolve_data_dir(data_dir)

    structs = resolve_subcortical_structs(structs)

    paths = load_dataset(
        data_dir=data_dir,
        subject_subset=resolve_subject_ids(subject_ids),
        struct_subset=structs,
        derivative=derivative,
        as_surface=False,
    )

    session_to_week = get_session_to_week_map(data_dir)

    return {
        get_subcortical_struct_long_name(struct): _load_struct_volumes(
            paths[struct], session_to_week
        )
        for struct in structs
    }
