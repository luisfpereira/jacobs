import os
from functools import partial
from pathlib import Path

import numpy as np
from polpo.enigma.mesh import load_template
from polpo.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.workflow.bcv import GroupedMeshRankSelectionRunner
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput


def prepare_inputs(struct, outputs_dir):
    output = LddmmToGlobalOutput(outputs_dir / struct)
    template = load_template(struct)

    dataset = output.decoded.global_deltas.map_values(
        lambda delta: np.asarray(delta).flatten()
    )

    return template.regular_faces, dataset


if __name__ == "__main__":
    outputs_dir = Path(os.environ["OUTPUTS_DIR"])

    struct_dirs = find_experiment_dirs(
        outputs_dir,
        long_name=False,
        interleave=True,
    )
    structs = list(struct_dirs.keys())

    prepare_inputs_ = partial(
        prepare_inputs,
        outputs_dir=outputs_dir,
    )

    seeds = [0, 1, 2, 3, 4]

    for n_parts in [3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
        for rep, seed in enumerate(seeds):
            # TODO: update manifest dir
            runner = GroupedMeshRankSelectionRunner(
                structs,
                prepare_inputs_,
                get_results_dir=lambda struct: outputs_dir
                / struct
                / "rank_selection_diagnostics"
                / f"n_parts_{n_parts}_rep_{rep}",
                n_parts=n_parts,
                n_groups=1,
                state_dir=outputs_dir
                / ".rank_selection"
                / f"n_parts_{n_parts}_rep_{rep}",
                center=True,
                seed=seed,
            )

            runner.run(overwrite=False)
