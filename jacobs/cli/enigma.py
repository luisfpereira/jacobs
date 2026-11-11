import click

from . import _options


@click.group()
def enigma():
    """Analyses using ENIGMA derivatives."""


@enigma.command()
@_options.figures_dir
@_options.runner_options
@_options.plotting
def volumes(figures_dir, overwrite, tasks, raise_on_error):
    """Generate ENIGMA volume figures."""
    from jacobs.core.style import get_subject_colors
    from jacobs.core.volumes.enigma import load_volumes
    from jacobs.core.volumes.figures import VolumeFiguresRunner

    figures_dir = figures_dir / "volumes"

    runner = VolumeFiguresRunner(
        figures_dir,
        load_volumes,
        colors=get_subject_colors(),
    )
    runner.run(
        overwrite=overwrite,
        tasks=tasks or None,
        raise_on_error=raise_on_error,
    )


@enigma.command()
@_options.outputs_dir
@_options.runner_options
def noise_floor(outputs_dir, overwrite, tasks, raise_on_error):
    """Compute distances between sessions of each control subject.

    The noise floor of a structure is the median, across controls, of the RMS
    distance between their sessions. It is measured with the varifold width
    shared by the Euclidean and LDDMM pipelines.
    """
    from functools import partial

    from polpo.workflow.task import CompositeTaskRunner, TaskRunner

    from jacobs.core.enigma import (
        get_noise_floor_dir,
        load_noise_floor_inputs,
        save_noise_floor_distances,
    )
    from jacobs.core.study import resolve_subcortical_structs
    from jacobs.core.varifold import instantiate_varifold_metric

    runners = {}
    for struct in resolve_subcortical_structs():
        controls, baseline_inputs = load_noise_floor_inputs(struct)
        metric = instantiate_varifold_metric(baseline_inputs.flatten().values_list())
        results_dir = get_noise_floor_dir(outputs_dir / struct)

        runners[struct] = TaskRunner.from_callable(
            partial(save_noise_floor_distances, results_dir, controls, metric),
            state_dir=results_dir,
            name="compute_noise_floor_distances",
        )

    runner = CompositeTaskRunner(runners, state_dir=outputs_dir / ".noise_floor")
    runner.run(overwrite=overwrite, tasks=tasks or None, raise_on_error=raise_on_error)


@enigma.command()
@_options.outputs_dir
@_options.figures_dir
@_options.overwrite
@_options.plotting
def noise_floor_figures(outputs_dir, figures_dir, overwrite):
    """Plot the per-subject distances between sessions of control subjects.

    Boxes show, per structure and side, the RMS distance of each control subject;
    their medians are the noise floors.
    """
    from functools import partial

    from jacobs.core.enigma import NoiseFloorFiguresRunner, load_all_subject_distances

    NoiseFloorFiguresRunner(
        figures_dir / "noise_floor",
        partial(load_all_subject_distances, outputs_dir),
    ).run(overwrite=overwrite)
