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
    )
