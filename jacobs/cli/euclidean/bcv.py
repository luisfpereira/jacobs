"""Bi-cross-validation of PCA ranks on Euclidean shape representations.

A full analysis runs, in order::

    jacobs euclidean bcv rank-selection
    jacobs euclidean bcv rank-selection-figures
    jacobs euclidean bcv reconstruction-eval
    jacobs euclidean bcv longitudinal-fidelity
    jacobs euclidean bcv evaluation-figures

``--features`` is ``deltas`` (baseline deltas) or ``vertices`` (registered
vertices). Use the same choice for every step, and one outputs directory per
choice.
"""

from functools import partial

import click

from jacobs.cli import _options


@click.group()
def bcv():
    """Bi-cross-validation rank selection on global deltas."""


@bcv.command()
@_options.outputs_dir
@_options.features
@_options.runner_options
def rank_selection(outputs_dir, features, overwrite, tasks, raise_on_error):
    """Select PCA rank per structure via bi-cross-validation."""
    from polpo.workflow.task import CompositeTaskRunner, TaskRunner

    from jacobs.core.bcv import get_rank_selection_dir
    from jacobs.core.euclidean.bcv import select_rank
    from jacobs.core.study import resolve_subcortical_structs

    runners = {
        struct: TaskRunner.from_callable(
            partial(select_rank, struct, outputs_dir / struct, features),
            state_dir=get_rank_selection_dir(outputs_dir / struct),
            name="select_rank",
        )
        for struct in resolve_subcortical_structs()
    }
    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".bcv" / "rank_selection",
    )
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=raise_on_error)


@bcv.command()
@_options.outputs_dir
@_options.figures_dir
@_options.overwrite
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
@_options.features
@_options.runner_options
def reconstruction_eval(outputs_dir, features, overwrite, tasks, raise_on_error):
    """Evaluate PCA reconstruction errors at each structure's BCV rank."""
    from polpo.surface_mesh.transform import DeltaTransform, VerticesTransform
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.bcv import get_reconstruction_eval_dir, load_selected_rank
    from jacobs.core.euclidean import compute_atlas
    from jacobs.core.euclidean import load_eval_meshes
    from jacobs.core.pca import PCAReconstructionEvaluator
    from jacobs.core.study import resolve_subcortical_structs
    from jacobs.core.varifold import instantiate_varifold_metric

    runners = {}
    for struct in resolve_subcortical_structs():
        struct_dir = outputs_dir / struct
        meshes, baseline_inputs = load_eval_meshes(struct, features)
        transform = (
            DeltaTransform(compute_atlas(baseline_inputs))
            if features == "deltas"
            else VerticesTransform(meshes.flatten().values_list()[0].faces)
        )
        metric = instantiate_varifold_metric(baseline_inputs.flatten().values_list())

        runners[struct] = PCAReconstructionEvaluator(
            get_reconstruction_eval_dir(struct_dir),
            meshes.flatten(),
            transform,
            metrics={"squared_varifold": metric.squared_dist},
            ks=[int(load_selected_rank(struct_dir))],
        )

    runner = CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".bcv" / "reconstruction_eval"
    )
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=raise_on_error)


@bcv.command()
@_options.outputs_dir
@_options.features
@_options.runner_options
def longitudinal_fidelity(outputs_dir, features, overwrite, tasks, raise_on_error):
    """Compare log-volume slopes of input and reconstructed meshes at the BCV rank."""
    from polpo.datasets.jacobs.utils import is_pregnancy_subject
    from polpo.surface_mesh.transform import DeltaTransform, VerticesTransform
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.bcv import get_longitudinal_fidelity_dir, load_selected_rank
    from jacobs.core.euclidean import compute_atlas
    from jacobs.core.euclidean import load_eval_meshes
    from jacobs.core.paths import resolve_data_dir
    from jacobs.core.pca import LongitudinalFidelityEvaluator
    from jacobs.core.study import get_session_to_week_map, resolve_subcortical_structs

    session_to_week = get_session_to_week_map(resolve_data_dir(None))

    runners = {}
    for struct in resolve_subcortical_structs():
        struct_dir = outputs_dir / struct
        meshes, baseline_inputs = load_eval_meshes(struct, features)
        transform = (
            DeltaTransform(compute_atlas(baseline_inputs))
            if features == "deltas"
            else VerticesTransform(meshes.flatten().values_list()[0].faces)
        )
        timed_meshes = meshes.map_keys(session_to_week)

        runners[struct] = LongitudinalFidelityEvaluator(
            get_longitudinal_fidelity_dir(struct_dir),
            timed_meshes,
            transform,
            ks=[int(load_selected_rank(struct_dir))],
            subjects=[s for s in timed_meshes.keys_list() if is_pregnancy_subject(s)],
        )

    runner = CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".bcv" / "longitudinal_fidelity"
    )
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=raise_on_error)


@bcv.command()
@_options.outputs_dir
@_options.figures_dir
@_options.runner_options
@click.option(
    "--legend",
    is_flag=True,
    help="Add a legend to each figure instead of a single merged one.",
)
@_options.noise_floor_dir
@_options.plotting
def evaluation_figures(
    outputs_dir, figures_dir, noise_floor_dir, legend, overwrite, tasks, raise_on_error
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
    ).run(overwrite=overwrite, tasks=tasks or None, raise_on_error=raise_on_error)
