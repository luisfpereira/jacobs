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
@_options.tasks
@click.option("--overwrite", is_flag=True)
def reconstruction_eval(outputs_dir, tasks, overwrite):
    """Evaluate PCA reconstruction error at each structure's BCV-selected rank."""
    from polpo.datasets.jacobs.utils import is_pregnancy_subject
    from polpo.surface_mesh.transform import DeltaTransform
    from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.bcv import (
        get_longitudinal_fidelity_dir,
        get_reconstruction_eval_dir,
        load_selected_rank,
    )
    from jacobs.core.lddmm.outputs import (
        find_experiment_dirs,
        load_global_atlas,
        load_global_meshes,
    )
    from jacobs.core.paths import resolve_data_dir
    from jacobs.core.pca import (
        LongitudinalFidelityEvaluator,
        PCAReconstructionEvaluator,
    )
    from jacobs.core.study import get_session_to_week_map

    session_to_week = get_session_to_week_map(resolve_data_dir(None))

    reconstruction = {}
    longitudinal = {}
    for struct, struct_dir in find_experiment_dirs(outputs_dir).items():
        output = LddmmToGlobalOutput(struct_dir)
        meshes = load_global_meshes(output)
        transform = DeltaTransform(load_global_atlas(output))
        ks = [int(load_selected_rank(struct_dir))]

        reconstruction[struct] = PCAReconstructionEvaluator(
            get_reconstruction_eval_dir(struct_dir),
            meshes.flatten(),
            transform,
            metrics={
                "squared_varifold": output.instantiate_varifold_metric().squared_dist,
            },
            ks=ks,
        )

        timed_meshes = meshes.map_keys(session_to_week)
        longitudinal[struct] = LongitudinalFidelityEvaluator(
            get_longitudinal_fidelity_dir(struct_dir),
            timed_meshes,
            transform,
            ks=ks,
            subjects=[s for s in timed_meshes.keys_list() if is_pregnancy_subject(s)],
        )

    runner = CompositeTaskRunner(
        {
            "reconstruction": CompositeTaskRunner(
                reconstruction, state_dir=outputs_dir / ".bcv" / "reconstruction_eval"
            ),
            "longitudinal": CompositeTaskRunner(
                longitudinal, state_dir=outputs_dir / ".bcv" / "longitudinal_fidelity"
            ),
        },
        state_dir=outputs_dir / ".bcv" / "evaluation",
    )
    runner.run(
        overwrite=overwrite,
        tasks=tasks or None,
        raise_on_error=True,
    )


@bcv.command()
@_options.outputs_dir
@_options.figures_dir
@_options.tasks
@click.option("--overwrite", is_flag=True)
@_options.plotting
def reconstruction_eval_figures(outputs_dir, figures_dir, tasks, overwrite):
    """Plot reconstruction errors and trend attenuation at the BCV rank.

    Tasks: plot_errors, plot_normalized_errors, plot_attenuation. Defaults to all.
    """
    from jacobs.core.bcv import ReconstructionEvalFiguresRunner
    from jacobs.core.lddmm.bcv import (
        load_longitudinal_fidelity_slopes,
        load_reconstruction_eval_errors,
    )
    from jacobs.core.lddmm.outputs import load_atlas_rms_distances

    ReconstructionEvalFiguresRunner(
        figures_dir / "bcv" / "reconstruction_eval",
        partial(load_reconstruction_eval_errors, outputs_dir),
        partial(load_atlas_rms_distances, outputs_dir),
        partial(load_longitudinal_fidelity_slopes, outputs_dir),
    ).run(
        tasks=tasks or None,
        overwrite=overwrite,
        raise_on_error=True,
    )
