from functools import partial

import click

from jacobs.cli import _options


@click.group()
def bcv():
    """Bi-cross-validation rank selection on global deltas."""


@bcv.command()
@_options.outputs_dir
@click.option("--overwrite", is_flag=True)
def rank_selection(outputs_dir, overwrite):
    """Select PCA rank per structure via bi-cross-validation on global deltas."""
    from functools import partial

    from polpo.workflow.task import CompositeTaskRunner, TaskRunner

    from jacobs.core.bcv import get_rank_selection_dir
    from jacobs.core.lddmm.bcv import select_rank
    from jacobs.core.lddmm.outputs import find_experiment_dirs

    runners = {
        struct: TaskRunner.from_callable(
            partial(select_rank, struct_dir),
            state_dir=get_rank_selection_dir(struct_dir),
            name="select_rank",
        )
        for struct, struct_dir in find_experiment_dirs(outputs_dir).items()
    }
    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".bcv" / "rank_selection",
    )
    runner.run(overwrite=overwrite)


@bcv.command()
@_options.outputs_dir
@_options.figures_dir
@click.option("--overwrite", is_flag=True)
@_options.plotting
def rank_selection_figures(outputs_dir, figures_dir, overwrite):
    """Plot rank-selection errors per structure."""
    from jacobs.core.bcv import RankSelectionFiguresRunner
    from jacobs.core.lddmm.bcv import load_rank_selection_results

    RankSelectionFiguresRunner(
        figures_dir / "bcv" / "rank_selection",
        partial(load_rank_selection_results, outputs_dir),
    ).run(overwrite=overwrite)


@bcv.command()
@_options.outputs_dir
@click.option("--overwrite", is_flag=True)
def reconstruction(outputs_dir, overwrite):
    """Evaluate PCA reconstruction error at each structure's BCV-selected rank."""
    from polpo.surface_mesh.transform import DeltaTransform
    from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.bcv import get_reconstruction_eval_dir, load_selected_rank
    from jacobs.core.lddmm.outputs import (
        find_experiment_dirs,
        load_global_atlas,
        load_global_meshes,
    )
    from jacobs.core.pca import PCAReconstructionEvaluator

    runners = {}
    for struct, struct_dir in find_experiment_dirs(outputs_dir).items():
        output = LddmmToGlobalOutput(struct_dir)

        runners[struct] = PCAReconstructionEvaluator(
            get_reconstruction_eval_dir(struct_dir),
            load_global_meshes(output).flatten(),
            DeltaTransform(load_global_atlas(output)),
            metrics={
                "squared_varifold": output.instantiate_varifold_metric().squared_dist,
            },
            ks=[load_selected_rank(struct_dir)],
        )

    CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".bcv" / "reconstruction_eval"
    ).run(overwrite=overwrite)
