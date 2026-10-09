import shutil
from pathlib import Path

import polpo.utils as putils
from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.workflow.lddmm_to_global.distances import (
    LddmmDistances,
    PersistentEvaluator,
)
from polpo.workflow.task import CompositeTaskRunner

if __name__ == "__main__":
    run_index = 3
    derivative = "enigma"

    name = "post_dists/lddmm"

    experiments_dir = (
        putils.get_frank_user_scratch()
        if putils.in_frank()
        else Path.home() / "experiments"
    )

    outputs_dir = (
        experiments_dir / f"jacobs_{derivative}" / f"lddmm_to_global_{run_index}"
    )

    experiment_dirs = find_experiment_dirs(outputs_dir)

    runners = {
        struct: PersistentEvaluator(
            LddmmDistances(experiment_dir),
            experiment_dir / name,
        ).resolve(
            tasks=(
                "local_reconstruction_error",
                "local_atlas_fit_error",
                "global_atlas_fit_error",
                "local_to_global_reconstruction_error",
                "local_to_global_transport_error",
                "local_atlas_distance",
                "global_atlas_distance",
            )
        )
        for struct, experiment_dir in experiment_dirs.items()
    }

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / f".{name}",
    )

    runner.run(
        overwrite=False,
        continue_on_error=True,
    )

    shutil.copy(Path(__file__).resolve(), outputs_dir / f"{name}.py")
