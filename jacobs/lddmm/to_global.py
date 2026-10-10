import shutil
from functools import partial
from pathlib import Path

import polpo.utils as putils
from polpo.datasets.jacobs.utils import get_subject_ids
from polpo.neuroimaging.enigma.naming import get_all_subcortical_structs
from polpo.workflow.lddmm_to_global import LddmmToGlobal
from polpo.workflow.task import CompositeTaskRunner, TaskRunner

from .core import prepare_inputs

DEBUG = False


def to_global(
    struct,
    subject_ids,
    data_dir,
    results_dir,
    derivative="enigma",
    **protocol_kwargs,
):
    dataset, atlas_keys, atlas_only_keys, known_correspondences, metadata = (
        prepare_inputs(struct, subject_ids, data_dir, derivative)
    )

    protocol = LddmmToGlobal(
        results_dir=results_dir,
        metadata=metadata,
        known_correspondences=known_correspondences,
        random_state=2026,
        **protocol_kwargs,
    )

    protocol.run(
        dataset,
        atlas_keys=atlas_keys,
        atlas_only_keys=atlas_only_keys,
    )


if __name__ == "__main__":
    run_index = 3

    if DEBUG:
        structs = ["L_Hipp"]
    else:
        structs = get_all_subcortical_structs(only_bilateral=True)
        structs.remove("L_Accu")
        structs.remove("R_Accu")

    subject_ids = get_subject_ids(sort=True, include_male=False)

    derivative = "enigma"

    data_dir = (
        Path("/scratch/data/maternal")
        if putils.in_frank()
        else Path.home() / "data/maternal"
    )

    outputs_dir = (
        putils.get_results_path()
        / f"jacobs_{derivative}"
        / f"lddmm_to_global_{run_index}"
    )

    runners = {
        struct: TaskRunner.from_callable(
            partial(
                to_global,
                struct,
                subject_ids,
                data_dir,
                results_dir=outputs_dir / struct,
                object_ratio=0.2,
                ratio_kernel=1.2,
            ),
            state_dir=outputs_dir / struct,
            name="to_global",
        )
        for struct in structs
    }

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".lddmm_to_global",
    )

    runner.run(
        overwrite=False,
    )

    shutil.copy(Path(__file__).resolve(), outputs_dir / "run.py")

    if runner.manifest.status == "failed":
        raise RuntimeError("All tasks failed.")
