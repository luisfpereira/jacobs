from functools import partial

from polpo.datasets.jacobs.mri import load_subcortical_segmentations
from polpo.neuroimaging.dataset import group_by_structure
from polpo.neuroimaging.mri import compute_segmentation_volumes
from polpo.neuroimaging.naming import get_subcortical_struct_long_name

from jacobs.core.paths import resolve_data_dir
from jacobs.core.study import (
    filter_pregnancy_weeks,
    get_session_to_week_map,
    group_by_pregnancy,
    resolve_subcortical_structs,
    resolve_subject_ids,
)


def load_volumes(structs=None, subject_ids=None, data_dir=None):
    derivative = "fast"

    data_dir = resolve_data_dir(data_dir)
    structs = resolve_subcortical_structs(structs)

    paths = load_subcortical_segmentations(
        derivative,
        data_dir=data_dir,
        subject_subset=resolve_subject_ids(subject_ids),
        as_image=False,
    )

    session_to_week = get_session_to_week_map(data_dir)
    paths = filter_pregnancy_weeks(paths, session_to_week).map_keys(session_to_week)

    # NB: slow
    volumes = group_by_structure(
        paths.map_values(
            partial(compute_segmentation_volumes, labels=structs, derivative=derivative)
        )
    )

    return {
        get_subcortical_struct_long_name(struct): group_by_pregnancy(volumes[struct])
        for struct in structs
    }
