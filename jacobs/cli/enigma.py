from pathlib import Path

import click

from jacobs.core import update_mpl_params


@click.group()
def enigma():
    """Analyses using ENIGMA derivatives."""


@enigma.command()
@click.option(
    "--figures-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="FIGURES_DIR",
    required=True,
)
@click.option("--subdir/--no-subdir", default=True)
@click.option("--overwrite", is_flag=True)
def volumes(figures_dir, subdir, overwrite):
    """Generate ENIGMA volume figures."""

    from jacobs.volumes.enigma import make_runner

    if subdir:
        figures_dir = figures_dir / "volumes"

    update_mpl_params()

    runner = make_runner(figures_dir)

    runner.run(
        overwrite=overwrite,
        tasks=(
            "longitudinal",
            "trajectory_summary",
            "trajectory_summary_lr",
        ),
    )
