from polpo.bcv.mesh import GroupedMeshRankSelection, GroupedMeshRankSelectionResult
from polpo.ext.numpy.io import load_indexed_arrays
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput

from jacobs.core.bcv import (
    get_longitudinal_fidelity_dir,
    get_rank_selection_dir,
    get_reconstruction_eval_dir,
)
from jacobs.core.pca import load_reconstruction_errors

from .outputs import (
    find_experiment_dirs,
    load_global_atlas,
    load_global_deltas,
)


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


def load_reconstruction_eval_errors(outputs_dir, metric="squared_varifold"):
    """Load BCV-rank squared errors as (rank, errors), keyed by long structure name."""
    errors = {}
    for struct, struct_dir in find_experiment_dirs(outputs_dir, long_name=True).items():
        ks, values = load_reconstruction_errors(
            get_reconstruction_eval_dir(struct_dir), metric
        )
        errors[struct] = (ks[0], values[:, 0])
    return errors


def load_longitudinal_fidelity_slopes(outputs_dir):
    """Load (rank, input slopes, reconstructed slopes) per experiment, keyed by long name."""
    slopes = {}
    for struct, struct_dir in find_experiment_dirs(outputs_dir, long_name=True).items():
        _, data = load_indexed_arrays(
            get_longitudinal_fidelity_dir(struct_dir) / "slopes.npz"
        )
        (rank,) = [k for k in data if k != "input"]
        slopes[struct] = (int(rank), data["input"], data[rank])

    return slopes
