from pathlib import Path

import polpo.utils as putils
from polpo.dataset import NestedKeyMap
from polpo.datasets.jacobs.mri import load_subcortical_segmentations
from polpo.datasets.jacobs.plot import get_subject_colors
from polpo.datasets.jacobs.tabular import get_session_to_week
from polpo.datasets.jacobs.utils import get_subject_ids, is_pregnancy_subject
from polpo.neuroimaging.dataset import group_by_structure
from polpo.neuroimaging.mri import compute_segmentation_volumes
from polpo.neuroimaging.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)

from .core import VolumeFiguresRunner


def prepare_inputs(subject_ids, data_dir, structs, derivative="fast"):
    """Compute longitudinal subcortical volumes from segmentations."""
    paths = load_subcortical_segmentations(
        derivative,
        data_dir=data_dir,
        subject_subset=subject_ids,
        as_image=False,
    )

    session_to_week = NestedKeyMap.from_inner_key_map(get_session_to_week())

    paths = paths.map_keys(session_to_week)

    paths = paths.filter_keys(
        lambda subject, week: (not is_pregnancy_subject(subject) or 0 <= week <= 42)
    )

    # NB: slow
    volumes = group_by_structure(
        paths.map_values(
            lambda path: compute_segmentation_volumes(
                path,
                labels=structs,
                derivative=derivative,
            )
        )
    )

    return {
        get_subcortical_struct_long_name(struct): volumes[struct].group_outer(
            lambda subject: "case" if is_pregnancy_subject(subject) else "control"
        )
        for struct in structs
    }


def make_runner(figures_dir):
    data_dir = (
        Path("/scratch/data/maternal")
        if putils.in_frank()
        else Path.home() / "data/maternal"
    )

    structs = get_all_subcortical_structs(
        only_bilateral=True,
        interleave=True,
    )
    structs.remove("L_Accu")
    structs.remove("R_Accu")

    subject_ids = get_subject_ids(sort=True, include_male=False)

    volumes = prepare_inputs(subject_ids, data_dir, structs)

    return VolumeFiguresRunner(
        figures_dir,
        volumes,
        colors=get_subject_colors(subject_ids),
    )
