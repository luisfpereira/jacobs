import logging

from polpo.dataset import NestedKeyMap
from polpo.datasets.jacobs.tabular import get_session_to_week
from polpo.datasets.jacobs.utils import get_subject_ids, is_pregnancy_subject
from polpo.neuroimaging.naming import get_all_subcortical_structs

logger = logging.getLogger(__name__)


def resolve_subject_ids(subject_ids=None):
    if subject_ids is None:
        return get_subject_ids(sort=True, include_male=False)

    return subject_ids


def resolve_subcortical_structs(structs=None):
    if structs is not None:
        return structs

    structs = get_all_subcortical_structs(
        only_bilateral=True,
        interleave=True,
    )
    structs.remove("L_Accu")
    structs.remove("R_Accu")

    return structs


def get_session_to_week_map(data_dir):
    return NestedKeyMap.from_inner_key_map(get_session_to_week(data_dir))


def filter_pregnancy_weeks(paths, session_to_week):
    """Restrict pregnancy observations to weeks 0-42, retaining all others."""

    def keep(subject, session):
        if not is_pregnancy_subject(subject):
            return True

        week = session_to_week.map_inner(subject, session)
        return 0 <= week <= 42

    return paths.filter_keys(keep)


def select_observations_and_atlas_inputs(dataset, session_to_week):
    """Select pregnancy-window observations and subject-atlas inputs.

    Subjects without suitable atlas inputs are dropped. Observations and atlas
    inputs may overlap (e.g. all control sessions are both).
    """
    atlas_inputs = select_atlas_inputs(dataset, session_to_week)

    missing = set(dataset.keys_list()) - set(atlas_inputs.keys_list())
    if missing:
        logger.info("Dropping subjects without suitable atlas inputs: %s", missing)

    observations = filter_pregnancy_weeks(dataset.drop_outer(missing), session_to_week)

    return observations, atlas_inputs


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


def group_by_pregnancy(dataset):
    return dataset.group_outer(
        lambda subject: "case" if is_pregnancy_subject(subject) else "control"
    )
