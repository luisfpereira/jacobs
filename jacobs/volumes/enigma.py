from pathlib import Path

import polpo.utils as putils
import pyvista as pv
from polpo.dataset import NestedKeyMap
from polpo.datasets.jacobs.mesh import load_dataset
from polpo.datasets.jacobs.plot import get_subject_colors
from polpo.datasets.jacobs.tabular import get_session_to_week
from polpo.datasets.jacobs.utils import get_subject_ids, is_pregnancy_subject
from polpo.neuroimaging.enigma.mesh import read_geometry
from polpo.neuroimaging.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)

from .core import VolumeFiguresRunner


def prepare_inputs(
    struct,
    subject_ids,
    data_dir,
    derivative="enigma",
):
    dataset = load_dataset(
        data_dir=data_dir,
        subject_subset=subject_ids,
        struct_subset=[struct],
        derivative=derivative,
        as_surface=False,
    )[struct]

    session_to_week = NestedKeyMap.from_inner_key_map(get_session_to_week())

    dataset = dataset.map_keys(session_to_week)

    grouped = dataset.group_outer(
        lambda subject: "case" if is_pregnancy_subject(subject) else "control"
    )

    # TODO: make this a proper function
    grouped = grouped.map_items(
        lambda key, data: (
            data.filter_keys(lambda _, time: 0 <= time <= 42) if key == "case" else data
        )
    )
    return grouped.map_values(
        lambda dataset_: dataset_.map_values(
            lambda path: pv.PolyData.from_regular_faces(*read_geometry(path)).volume
        )
    )


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

    volumes = {
        get_subcortical_struct_long_name(struct): prepare_inputs(
            struct, subject_ids, data_dir
        )
        for struct in structs
    }

    return VolumeFiguresRunner(
        figures_dir,
        volumes,
        colors=get_subject_colors(subject_ids),
    )
