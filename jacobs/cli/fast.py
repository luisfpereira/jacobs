import click

from . import _options


@click.group()
def fast():
    """Analyses using FAST derivatives."""


@fast.command()
@_options.figures_dir
@_options.runner_options
@_options.plotting
def volumes(figures_dir, overwrite, tasks, raise_on_error):
    """Generate FAST volume figures."""
    from jacobs.core.style import get_subject_colors
    from jacobs.core.volumes.fast import load_volumes
    from jacobs.core.volumes.figures import VolumeFiguresRunner

    runner = VolumeFiguresRunner(
        figures_dir / "volumes",
        load_volumes,
        colors=get_subject_colors(),
    )

    runner.run(
        overwrite=overwrite,
        tasks=tasks or None,
        raise_on_error=raise_on_error,
    )
