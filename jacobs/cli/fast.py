import click

from . import _options


@click.group()
def fast():
    """Analyses using FAST derivatives."""


@fast.command()
@_options.figures_dir
@click.option("--subdir/--no-subdir", default=True)
@click.option("--overwrite", is_flag=True)
@_options.tasks
@_options.plotting
def volumes(figures_dir, subdir, overwrite, tasks):
    """Generate FAST volume figures."""
    from jacobs.core.style import get_subject_colors
    from jacobs.core.volumes.fast import load_volumes
    from jacobs.core.volumes.figures import VolumeFiguresRunner

    if subdir:
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
