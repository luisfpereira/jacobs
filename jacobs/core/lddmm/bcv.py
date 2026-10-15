from polpo.bcv.mesh import GroupedMeshRankSelection, GroupedMeshRankSelectionResult
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput

from jacobs.core.bcv import get_rank_selection_dir

from .outputs import find_experiment_dirs, load_global_atlas, load_global_deltas


def select_rank(struct_dir):
    """Select the PCA rank of one experiment's global deltas via BCV and save it."""
    output = LddmmToGlobalOutput(struct_dir)

    selection = GroupedMeshRankSelection(center=True).fit(
        load_global_atlas(output).faces,
        load_global_deltas(output),
    )
    selection.result_.to_dir(get_rank_selection_dir(struct_dir))


def load_rank_selection_results(outputs_dir):
    """Load rank-selection results per experiment, keyed by long structure name."""
    return {
        struct: GroupedMeshRankSelectionResult.from_dir(
            get_rank_selection_dir(struct_dir)
        )
        for struct, struct_dir in find_experiment_dirs(
            outputs_dir, long_name=True
        ).items()
    }
