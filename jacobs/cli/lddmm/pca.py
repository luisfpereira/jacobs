from functools import partial

import click

from jacobs.cli import _options


@click.group()
def pca():
    """PCA of global deltas."""


@pca.command()
@_options.outputs_dir
@_options.figures_dir
@click.option("--n-components", type=click.IntRange(min=2), default=5)
@_options.runner_options
@_options.plotting
def pairplots(outputs_dir, figures_dir, n_components, overwrite, tasks, raise_on_error):
    """Generate PCA pairplots across subcortical structures."""
    from jacobs.core.lddmm.outputs import load_global_deltas_by_struct
    from jacobs.core.pca.plot import PCAPairplotsRunner
    from jacobs.core.style import get_subject_colors

    figures_dir = figures_dir / f"pca_{n_components}"

    runner = PCAPairplotsRunner(
        figures_dir,
        partial(load_global_deltas_by_struct, outputs_dir),
        colors=get_subject_colors(),
        n_components=n_components,
    )

    runner.run(
        overwrite=overwrite,
        tasks=tasks or None,
        raise_on_error=raise_on_error,
    )


@pca.command()
@_options.outputs_dir
@_options.runner_options
def reconstruction_eval(outputs_dir, overwrite, tasks, raise_on_error):
    """Evaluate PCA reconstruction error across a sweep of component counts."""
    from polpo.surface_mesh.transform import DeltaTransform
    from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.lddmm.outputs import (
        find_experiment_dirs,
        load_global_atlas,
        load_global_meshes,
    )
    from jacobs.core.pca import PCAReconstructionEvaluator

    ks = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 60, 80]

    runners = {}
    for struct, struct_dir in find_experiment_dirs(outputs_dir).items():
        output = LddmmToGlobalOutput(struct_dir)

        runners[struct] = PCAReconstructionEvaluator(
            struct_dir / "pca" / "reconstruction_eval",
            load_global_meshes(output).flatten(),
            DeltaTransform(load_global_atlas(output)),
            metrics={
                "squared_euclidean": output.instantiate_euclidean_metric().squared_dist,
                "squared_varifold": output.instantiate_varifold_metric().squared_dist,
            },
            ks=ks,
        )

    CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".pca" / "reconstruction_eval"
    ).run(
        overwrite=overwrite,
        tasks=tasks or None,
        raise_on_error=raise_on_error,
    )


@pca.command()
@_options.outputs_dir
@_options.figures_dir
@_options.runner_options
@_options.plotting
def reconstruction_eval_figures(
    outputs_dir, figures_dir, overwrite, tasks, raise_on_error
):
    """Plot PCA reconstruction errors, explained variance and BCV rank."""
    from jacobs.core.lddmm.pca import load_reconstruction_summaries
    from jacobs.core.pca.plot import PCAReconstructionFiguresRunner

    PCAReconstructionFiguresRunner(
        figures_dir / "pca" / "reconstruction_eval",
        partial(load_reconstruction_summaries, outputs_dir),
    ).run(
        overwrite=overwrite,
        tasks=tasks or None,
        raise_on_error=raise_on_error,
    )
