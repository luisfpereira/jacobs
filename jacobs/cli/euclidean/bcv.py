from functools import partial

import click

from jacobs.cli import _options


@click.group()
def bcv():
    """Bi-cross-validation rank selection on global deltas."""


@bcv.command()
@_options.outputs_dir
@_options.runner_options
def rank_selection(outputs_dir, overwrite, tasks, raise_on_error):
    """Select PCA rank per structure via bi-cross-validation on baseline deltas."""
    from polpo.workflow.task import CompositeTaskRunner, TaskRunner

    from jacobs.core.bcv import get_rank_selection_dir
    from jacobs.core.euclidean.bcv import select_rank
    from jacobs.core.study import resolve_subcortical_structs

    runners = {
        struct: TaskRunner.from_callable(
            partial(select_rank, struct, outputs_dir / struct),
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
@_options.runner_options
def reconstruction_eval(outputs_dir, overwrite, tasks, raise_on_error):
    """Evaluate PCA reconstruction error at each structure's BCV-selected rank."""
    from polpo.datasets.jacobs.utils import is_pregnancy_subject
    from polpo.surface_mesh.transform import DeltaTransform
    from polpo.surface_mesh.varifold.geometry import VarifoldMetric
    from polpo.surface_mesh.varifold.tuning import SigmaFromScale
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.bcv import (
        get_longitudinal_fidelity_dir,
        get_reconstruction_eval_dir,
        load_selected_rank,
    )
    from jacobs.core.euclidean import load_global_meshes
    from jacobs.core.paths import resolve_data_dir
    from jacobs.core.pca import (
        LongitudinalFidelityEvaluator,
        PCAReconstructionEvaluator,
    )
    from jacobs.core.study import get_session_to_week_map, resolve_subcortical_structs

    session_to_week = get_session_to_week_map(resolve_data_dir(None))

    reconstruction = {}
    longitudinal = {}
    for struct in resolve_subcortical_structs():
        struct_dir = outputs_dir / struct
        atlas, meshes = load_global_meshes(struct)
        transform = DeltaTransform(atlas)
        ks = [int(load_selected_rank(struct_dir))]

        sigma = float(SigmaFromScale(object_ratio=0.2)(meshes.flatten().values_list()))
        metric = VarifoldMetric(sigma=sigma, engine="auto", device="cpu")

        reconstruction[struct] = PCAReconstructionEvaluator(
            get_reconstruction_eval_dir(struct_dir),
            meshes.flatten(),
            transform,
            metrics={"squared_varifold": metric.squared_dist},
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
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=raise_on_error)


@bcv.command()
@_options.outputs_dir
@_options.figures_dir
@_options.runner_options
@_options.plotting
def reconstruction_eval_figures(
    outputs_dir, figures_dir, overwrite, tasks, raise_on_error
):
    """Plot reconstruction errors and trend attenuation at the BCV rank.

    Tasks: plot_errors, plot_normalized_errors, plot_attenuation. Defaults to all.
    """
    from jacobs.core.bcv import ReconstructionEvalFiguresRunner
    from jacobs.core.euclidean import compute_atlas_rms_distances
    from jacobs.core.lddmm.bcv import (
        load_longitudinal_fidelity_slopes,
        load_reconstruction_eval_errors,
    )

    ReconstructionEvalFiguresRunner(
        figures_dir / "bcv" / "reconstruction_eval",
        partial(load_reconstruction_eval_errors, outputs_dir),
        compute_atlas_rms_distances,
        partial(load_longitudinal_fidelity_slopes, outputs_dir),
    ).run(
        overwrite=overwrite,
        tasks=tasks or None,
        raise_on_error=raise_on_error,
    )
