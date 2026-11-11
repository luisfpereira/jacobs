from polpo.bcv.mesh import GroupedMeshRankSelection
from polpo.neuroimaging.enigma.mesh import load_template

from jacobs.core.bcv import get_rank_selection_dir

from jacobs.core.enigma import load_registered_dataset_with_baselines

from . import _check_features, compute_baseline_deltas, flatten_vertices


def load_rank_selection_features(
    struct, features="deltas", data_dir=None, subject_ids=None
):
    """Load the PCA features used for rank selection.

    Uses the same observations as ``load_eval_meshes``, so the selected rank
    applies to the evaluated meshes.

    Parameters
    ----------
    struct : str
        Subcortical structure.
    features : str
        ``"deltas"`` (observations minus each subject's mean baseline) or
        ``"vertices"`` (registered vertex coordinates).
    data_dir : pathlib.Path or None
        Data directory.
    subject_ids : sequence or None
        Subjects to load. If None, all subjects.

    Returns
    -------
    features : NestedDataset
        Feature vectors, keyed by subject and session.
    """
    _check_features(features)

    observations, baseline_inputs = load_registered_dataset_with_baselines(
        struct, data_dir, subject_ids
    )
    if features == "deltas":
        return compute_baseline_deltas(observations, baseline_inputs)
    return flatten_vertices(observations)


def select_rank(struct, struct_dir, features="deltas", data_dir=None, subject_ids=None):
    """Select the PCA rank of one structure's features via BCV and save it."""
    _, faces = load_template(struct, as_surface=False)

    selection = GroupedMeshRankSelection(center=True).fit(
        faces,
        load_rank_selection_features(struct, features, data_dir, subject_ids),
    )
    selection.result_.to_dir(get_rank_selection_dir(struct_dir))
