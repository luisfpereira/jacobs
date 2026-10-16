from polpo.bcv.mesh import GroupedMeshRankSelection
from polpo.neuroimaging.enigma.mesh import load_template

from jacobs.core.bcv import get_rank_selection_dir

from . import load_struct_features


def select_rank(struct, struct_dir, data_dir=None, subject_ids=None):
    """Select the PCA rank of one structure's baseline deltas via BCV and save it."""
    _, faces = load_template(struct, as_surface=False)

    selection = GroupedMeshRankSelection(center=True).fit(
        faces,
        load_struct_features(struct, data_dir, subject_ids),
    )
    selection.result_.to_dir(get_rank_selection_dir(struct_dir))
