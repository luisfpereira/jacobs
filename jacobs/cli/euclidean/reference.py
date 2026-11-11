"""Reference quantities for Euclidean shape representations.

::

    jacobs euclidean reference atlas-distance
"""

from functools import partial

import click

from jacobs.cli import _options


@click.group()
def reference():
    """Reference quantities for Euclidean shape representations."""


@reference.command()
@_options.outputs_dir
@_options.features
@_options.runner_options
def atlas_distance(outputs_dir, features, overwrite, tasks, raise_on_error):
    """Compute distances from each structure's meshes to the Euclidean atlas."""
    from polpo.workflow.task import CompositeTaskRunner, TaskRunner

    from jacobs.core.euclidean import (
        compute_atlas,
        get_atlas_distance_dir,
        load_eval_meshes,
        save_atlas_distances,
    )
    from jacobs.core.study import resolve_subcortical_structs
    from jacobs.core.varifold import instantiate_varifold_metric

    runners = {}
    for struct in resolve_subcortical_structs():
        meshes, baseline_inputs = load_eval_meshes(struct, features)
        atlas = compute_atlas(baseline_inputs)
        metric = instantiate_varifold_metric(baseline_inputs.flatten().values_list())
        results_dir = get_atlas_distance_dir(outputs_dir / struct)

        runners[struct] = TaskRunner.from_callable(
            partial(
                save_atlas_distances,
                results_dir,
                atlas,
                meshes.flatten(),
                metric,
            ),
            state_dir=results_dir,
            name="compute_atlas_distances",
        )

    runner = CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".reference" / "atlas_distance"
    )
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=raise_on_error)
