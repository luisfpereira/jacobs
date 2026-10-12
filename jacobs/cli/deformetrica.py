import click

from . import _options


@click.group()
@_options.log_level
def cli(log_level):
    """Jacobs analyses requiring the Deformetrica environment."""
    from jacobs.core.log import configure_logging

    configure_logging(log_level)


@cli.command()
@_options.outputs_dir
@click.option("--overwrite", is_flag=True)
def to_global(outputs_dir, overwrite):
    """Run LDDMM-to-global for each subcortical structure."""
    from functools import partial

    from polpo.workflow.task import CompositeTaskRunner, TaskRunner

    from jacobs.core.lddmm.to_global import run_to_global
    from jacobs.core.study import resolve_subcortical_structs

    runners = {
        struct: TaskRunner.from_callable(
            partial(
                run_to_global,
                struct,
                outputs_dir / struct,
                object_ratio=0.2,
                ratio_kernel=1.2,
            ),
            state_dir=outputs_dir / struct,
            name="to_global",
        )
        for struct in resolve_subcortical_structs()
    }

    runner = CompositeTaskRunner(runners, state_dir=outputs_dir / ".lddmm_to_global")
    runner.run(overwrite=overwrite)

    if runner.manifest.status == "failed":
        raise click.ClickException("All tasks failed.")


@cli.command()
@_options.outputs_dir
@click.option("--overwrite", is_flag=True)
def post_dists(outputs_dir, overwrite):
    """Compute LDDMM distances and errors for global representations."""
    from polpo.workflow.lddmm_to_global.distances import (
        LddmmDistances,
        PersistentEvaluator,
    )
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.lddmm import find_experiment_dirs

    tasks = (
        "local_reconstruction_error",
        "local_atlas_fit_error",
        "global_atlas_fit_error",
        "local_to_global_reconstruction_error",
        "local_to_global_transport_error",
        "local_atlas_distance",
        "global_atlas_distance",
    )

    runners = {
        struct: PersistentEvaluator(
            LddmmDistances(experiment_dir),
            experiment_dir / "post_dists" / "lddmm",
        ).resolve(tasks=tasks)
        for struct, experiment_dir in find_experiment_dirs(outputs_dir).items()
    }

    runner = CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".post_dists" / "lddmm"
    )
    runner.run(overwrite=overwrite)
