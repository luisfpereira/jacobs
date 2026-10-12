from functools import partial

import click

from . import _options


@click.group()
def euclidean():
    """Analyses using Euclidean."""


@euclidean.command()
@_options.figures_dir
@click.option("--subdir/--no-subdir", default=True)
@click.option("--n-components", type=click.IntRange(min=2), default=5)
@click.option("--overwrite", is_flag=True)
@click.option(
    "--features",
    type=click.Choice(["deltas", "vertices"]),
    multiple=True,
    default=("deltas", "vertices"),
    show_default=True,
)
@_options.plotting
def pca_pairplots(figures_dir, subdir, n_components, overwrite, features):
    """Generate PCA pairplots for baseline deltas and/or raw vertices."""
    from jacobs.core.euclidean import load_features
    from jacobs.core.pca import PCAPairplotsRunner
    from jacobs.core.style import get_subject_colors

    if subdir:
        figures_dir = figures_dir / f"pca_{n_components}"

    for kind in features:
        load = partial(load_features, subtract_baseline=kind == "deltas")

        PCAPairplotsRunner(
            figures_dir,
            load,
            colors=get_subject_colors(),
            n_components=n_components,
            prefix=kind,
        ).run(overwrite=overwrite)
