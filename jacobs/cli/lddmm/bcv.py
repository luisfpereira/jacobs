"""Bi-cross-validation of PCA ranks on LDDMM-to-global outputs.

A full analysis runs, in order, with ``OUTPUTS_DIR`` and ``FIGURES_DIR`` set
(e.g. ``source config/local/lddmm_enigma.sh``)::

    jacobs lddmm bcv rank-selection
    jacobs lddmm bcv rank-selection-figures
    jacobs lddmm bcv reconstruction-eval
    jacobs lddmm bcv longitudinal-fidelity
    jacobs lddmm bcv evaluation-figures
"""

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
    from jacobs.core.bcv import (
        RankSelectionFiguresRunner,
        get_rank_selection_figures_dir,
        load_rank_selection_results,
    )

    RankSelectionFiguresRunner(
        get_rank_selection_figures_dir(figures_dir),
        partial(load_rank_selection_results, outputs_dir),
    ).run(overwrite=overwrite)


@bcv.command()
@_options.outputs_dir
@_options.tasks
@click.option("--overwrite", is_flag=True)
def reconstruction_eval(outputs_dir, tasks, overwrite):
    """Evaluate PCA reconstruction errors at each structure's BCV-selected rank."""
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
            ks=[int(load_selected_rank(struct_dir))],
        )

    runner = CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".bcv" / "reconstruction_eval"
    )
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=True)


@bcv.command()
@_options.outputs_dir
@_options.tasks
@click.option("--overwrite", is_flag=True)
def longitudinal_fidelity(outputs_dir, tasks, overwrite):
    """Compare log-volume slopes of input and reconstructed meshes at the BCV rank."""
    from polpo.datasets.jacobs.utils import is_pregnancy_subject
    from polpo.surface_mesh.transform import DeltaTransform
    from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.bcv import get_longitudinal_fidelity_dir, load_selected_rank
    from jacobs.core.lddmm.outputs import (
        find_experiment_dirs,
        load_global_atlas,
        load_global_meshes,
    )
    from jacobs.core.paths import resolve_data_dir
    from jacobs.core.pca import LongitudinalFidelityEvaluator
    from jacobs.core.study import get_session_to_week_map

    session_to_week = get_session_to_week_map(resolve_data_dir(None))

    runners = {}
    for struct, struct_dir in find_experiment_dirs(outputs_dir).items():
        output = LddmmToGlobalOutput(struct_dir)
        timed_meshes = load_global_meshes(output).map_keys(session_to_week)

        runners[struct] = LongitudinalFidelityEvaluator(
            get_longitudinal_fidelity_dir(struct_dir),
            timed_meshes,
            DeltaTransform(load_global_atlas(output)),
            ks=[int(load_selected_rank(struct_dir))],
            subjects=[s for s in timed_meshes.keys_list() if is_pregnancy_subject(s)],
        )

    runner = CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".bcv" / "longitudinal_fidelity"
    )
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=True)


@bcv.command()
@_options.outputs_dir
@_options.figures_dir
@_options.tasks
@click.option("--overwrite", is_flag=True)
@click.option(
    "--legend",
    is_flag=True,
    help="Add a legend to each figure instead of a single merged one.",
)
@_options.noise_floor_dir
@_options.plotting
def evaluation_figures(
    outputs_dir, figures_dir, noise_floor_dir, tasks, legend, overwrite
):
    """Plot reconstruction errors and trend scaling factor at the BCV rank.

    Needs the results of reconstruction-eval and longitudinal-fidelity.

    Tasks: plot_errors, plot_scaling, plot_legend, merge_errors_scaling.
    Defaults to all.
    """
    from jacobs.core.bcv import EvaluationFiguresRunner, get_evaluation_figures_dir
    from jacobs.core.enigma import load_noise_floors
    from jacobs.core.lddmm.bcv import (
        load_longitudinal_fidelity_slopes,
        load_reconstruction_eval_errors,
    )

    EvaluationFiguresRunner(
        get_evaluation_figures_dir(figures_dir),
        partial(load_reconstruction_eval_errors, outputs_dir),
        partial(load_longitudinal_fidelity_slopes, outputs_dir),
        load_noise_floors=None
        if noise_floor_dir is None
        else partial(load_noise_floors, noise_floor_dir),
        legend=legend,
    ).run(
        tasks=tasks or None,
        overwrite=overwrite,
        raise_on_error=True,
    )
