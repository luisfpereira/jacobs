import logging
import os
from pathlib import Path

import numpy as np
from polpo.enigma.mesh import load_template
from polpo.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.workflow.bcv import GroupedMeshRankSelectionRunner
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
from polpo.workflow.task import CompositeTaskRunner


def prepare_inputs(struct, struct_dir):
    output = LddmmToGlobalOutput(struct_dir)
    _, faces = load_template(struct, as_polydata=False)

    dataset = output.decoded.global_deltas.map_values(
        lambda delta: np.asarray(delta).flatten()
    )

    return faces, dataset


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    outputs_dir = Path(os.environ["OUTPUTS_DIR"])

    struct_dirs = find_experiment_dirs(
        outputs_dir,
        long_name=False,
    )

    runners = {
        struct: GroupedMeshRankSelectionRunner.from_data(
            *prepare_inputs(
                struct,
                struct_dir,
            ),
            results_dir=struct_dir / "rank_selection",
            center=True,
        )
        for struct, struct_dir in struct_dirs.items()
    }

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".rank_selection",
    )

    runner.run(
        overwrite=True,
    )
