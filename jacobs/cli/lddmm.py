from pathlib import Path

import click

from jacobs.core import update_mpl_params


@click.group()
def lddmm():
    """Analyses using LDDMM."""


@lddmm.command()
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
    from jacobs.lddmm.pca_pairplot import make_runner

    if subdir:
        figures_dir = figures_dir / "pca"

    update_mpl_params()

    runner = make_runner(
        outputs_dir=outputs_dir,
        figures_dir=figures_dir,
        n_components=n_components,
    )
    runner.run(overwrite=overwrite)
