from functools import partial

import click

from jacobs.cli import _options


@click.group()
def pca():
    """PCA of global deltas."""


@pca.command()
@_options.outputs_dir
@_options.figures_dir
@click.option("--subdir/--no-subdir", default=True)
@click.option("--n-components", type=click.IntRange(min=2), default=5)
@click.option("--overwrite", is_flag=True)
@_options.plotting
def pca_pairplots(outputs_dir, figures_dir, subdir, n_components, overwrite):
    """Generate PCA pairplots across subcortical structures."""
    from jacobs.core.lddmm.outputs import load_global_deltas_by_struct
    from jacobs.core.pca import PCAPairplotsRunner
    from jacobs.core.style import get_subject_colors

    if subdir:
        figures_dir = figures_dir / f"pca_{n_components}"

    runner = PCAPairplotsRunner(
        figures_dir,
        partial(load_global_deltas_by_struct, outputs_dir),
        colors=get_subject_colors(),
        n_components=n_components,
    )
    runner.run(overwrite=overwrite)


@pca.command()
@_options.outputs_dir
@click.option("--overwrite", is_flag=True)
def pca_reconstruction_eval(outputs_dir, overwrite):
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
    ).run(overwrite=overwrite)


@pca.command()
@_options.outputs_dir
@_options.figures_dir
@click.option("--overwrite", is_flag=True)
@_options.plotting
def pca_reconstruction_eval_figures(outputs_dir, figures_dir, overwrite):
    """Plot PCA reconstruction errors, explained variance and BCV rank."""
    from jacobs.core.lddmm.pca import load_reconstruction_summaries
    from jacobs.core.pca import PCAReconstructionFiguresRunner

    PCAReconstructionFiguresRunner(
        figures_dir / "pca" / "reconstruction_eval",
        partial(load_reconstruction_summaries, outputs_dir),
    ).run(
        overwrite=overwrite,
        raise_on_error=True,
    )
