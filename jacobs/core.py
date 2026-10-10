from pathlib import Path

import matplotlib as mpl
import polpo.utils as putils
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from polpo.dataset import NestedKeyMap
from polpo.datasets.jacobs.tabular import get_session_to_week
from polpo.datasets.jacobs.utils import get_subject_ids, is_pregnancy_subject


def get_data_dir():
    return (
        Path("/scratch/data/maternal")
        # TODO: move frank from polpo?
        if putils.in_frank()
        else Path.home() / "data/maternal"
    )


def resolve_data_dir(data_dir=None):
    if data_dir is None:
        return get_data_dir()

    return data_dir


def resolve_subject_ids(subject_ids=None):
    if subject_ids is None:
        return get_subject_ids(sort=True, include_male=False)

    return subject_ids


def update_mpl_params(format="pdf"):
    mpl.rcParams.update(
        {
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "savefig.format": "pdf",
            "image.cmap": "bwr",
            "pdf.fonttype": 42,
        }
    )


def get_subject_colors(subj_ids=None):
    if subj_ids is None:
        subj_ids = get_subject_ids(include_male=False, sort=True)

    color_ids = [
        subj_id for subj_id in subj_ids if not str(subj_id).startswith(("3", "4"))
    ]

    tab10 = plt.colormaps["tab10"]
    base_colors = [tab10(i) for i in range(tab10.N) if i != 7]

    cmap = mcolors.ListedColormap(base_colors).resampled(len(color_ids))

    colors = dict(zip(color_ids, cmap(range(len(color_ids)))))
    colors.update(
        {subj_id: "gray" for subj_id in subj_ids if str(subj_id).startswith(("3", "4"))}
    )
    return colors


def get_session_to_week_map():
    return NestedKeyMap.from_inner_key_map(get_session_to_week())


def filter_pregnancy_weeks(paths, session_to_week=None):
    """Restrict pregnancy observations to weeks 0-42, retaining all others."""

    if session_to_week is None:
        session_to_week = get_session_to_week_map()

    def keep(subject, session):
        if not is_pregnancy_subject(subject):
            return True

        week = session_to_week.map_inner(subject, session)
        return 0 <= week <= 42

    return paths.filter_keys(keep)


def select_atlas_inputs(dataset, session_to_week):
    """Select observations used to estimate subject-specific atlases.

    For pregnancy subjects, retain only the last available session when it
    occurs after gestational week 42. For non-pregnancy subjects, retain all
    available observations.

    Parameters
    ----------
    dataset : NestedDataset
        Dataset indexed by subject and session.
    session_to_week : NestedKeyMap
        Mapping from subject-session keys to gestational week.

    Returns
    -------
    atlas_inputs : NestedDataset
        Observations to use for atlas estimation.
    """
    last_session = {
        subject: sessions[-1] for subject, sessions in dataset.nested_keys().items()
    }

    def use_for_atlas(subject, session):
        if not is_pregnancy_subject(subject):
            return True

        return (
            session == last_session[subject]
            and session_to_week.map_inner(subject, session) > 42.0
        )

    return dataset.filter_keys(use_for_atlas)
