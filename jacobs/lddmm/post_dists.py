import shutil
from pathlib import Path

import polpo.utils as putils
from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.workflow.lddmm_to_global.distances import (
    EuclideanDistances,
    PersistentEvaluator,
    VarifoldDistances,
)
from polpo.workflow.task import CompositeTaskRunner

if __name__ == "__main__":
    run_index = 3
    derivative = "enigma"

    metrics = {
        "eucl": EuclideanDistances,
        "var": VarifoldDistances,
    }

    name = "post_dists"

    experiments_dir = (
        putils.get_frank_user_scratch()
        if putils.in_frank()
        else Path.home() / "experiments"
    )

    outputs_dir = (
        experiments_dir / f"jacobs_{derivative}" / f"lddmm_to_global_{run_index}"
    )

    experiment_dirs = find_experiment_dirs(outputs_dir)

    runners = {}
    for metric, Evaluator in metrics.items():
        runners_ = {
            struct: PersistentEvaluator(
                Evaluator(experiment_dir),
                experiment_dir / f"post_dists/{metric}",
            )
            for struct, experiment_dir in experiment_dirs.items()
        }

        runners[metric] = CompositeTaskRunner(
            runners_,
            state_dir=outputs_dir / f".{name}/{metric}",
        )

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / f".{name}",
    )

    runner.run(
        overwrite=False,
        continue_on_error=True,
    )

    shutil.copy(Path(__file__).resolve(), outputs_dir / f"{name}.py")
