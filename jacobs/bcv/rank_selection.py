import logging
import os
from functools import partial
from pathlib import Path

import numpy as np

from polpo.bcv.mesh import GroupedMeshRankSelection
from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.neuroimaging.enigma.mesh import load_template
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
from polpo.workflow.task import CompositeTaskRunner, TaskRunner


def prepare_inputs(struct, struct_dir):
    output = LddmmToGlobalOutput(struct_dir)
    template = load_template(struct)

    dataset = output.view.global_deltas.map_values(
        lambda delta: np.asarray(delta).flatten()
    )

    return template.faces, dataset


def select_rank(struct, struct_dir, selector, results_dir):
    mesh_faces, dataset = prepare_inputs(struct, struct_dir)

    selection = selector.fit(mesh_faces, dataset)

    selection.result_.to_dir(results_dir)


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
        struct: TaskRunner.from_callable(
            partial(
                select_rank,
                struct,
                struct_dir,
                GroupedMeshRankSelection(center=True),
                results_dir=struct_dir / "rank_selection",
            ),
            state_dir=struct_dir / "rank_selection",
            name="select_rank",
        )
        for struct, struct_dir in struct_dirs.items()
    }

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".rank_selection",
    )

    runner.run(
        overwrite=True,
        # tasks=["L_Hipp"],
    )
