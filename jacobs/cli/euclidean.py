from pathlib import Path

import click

from jacobs.core import update_mpl_params


@click.group()
def euclidean():
    """Analyses using Euclidean."""


@euclidean.command()
@click.option(
    "--figures-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="FIGURES_DIR",
    required=True,
)
@click.option("--subdir/--no-subdir", default=True)
@click.option("--n-components", type=click.IntRange(min=2), default=5)
@click.option("--overwrite", is_flag=True)
@click.option("--vertices/--deltas", default=False)
def pca_pairplots(figures_dir, subdir, n_components, overwrite, vertices):
    """Generate PCA pairplots across subcortical structures."""
    from jacobs.euclidean.pca_pairplot import make_runner

    if subdir:
        figures_dir = figures_dir / "pca"
        if vertices:
            figures_dir = figures_dir / "vertices"

    update_mpl_params()

    runner = make_runner(
        figures_dir=figures_dir,
        n_components=n_components,
        subtract_baseline=not vertices,
    )
    runner.run(overwrite=overwrite)
