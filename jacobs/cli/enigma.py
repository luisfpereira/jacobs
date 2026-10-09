from pathlib import Path

import click
from polpo.datasets.jacobs.plot import update_mpl_params


@click.group()
def cli():
    """Jacobs dataset analyses."""


@cli.group()
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


@enigma.command()
@click.option(
    "--outputs-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="OUTPUTS_DIR",
    required=True,
)
@click.option(
    "--figures-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="FIGURES_DIR",
    required=True,
)
@click.option("--subdir/--no-subdir", default=True)
@click.option("--n-components", type=click.IntRange(min=2), default=5)
@click.option("--overwrite", is_flag=True)
def pca_pairplots(outputs_dir, figures_dir, subdir, n_components, overwrite):
    """Generate PCA pairplots across subcortical structures."""
    from jacobs.pca.pairplot import make_runner

    if subdir:
        figures_dir = figures_dir / "pca"

    update_mpl_params()

    runner = make_runner(
        outputs_dir=outputs_dir,
        figures_dir=figures_dir,
        n_components=n_components,
    )
    runner.run(overwrite=overwrite)
