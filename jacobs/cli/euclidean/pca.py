from functools import partial

import click

from jacobs.cli import _options


@click.group()
def pca():
    """PCA of global deltas."""


@pca.command()
@_options.figures_dir
@click.option("--n-components", type=click.IntRange(min=2), default=5)
@click.option(
    "--features",
    type=click.Choice(["deltas", "vertices"]),
    multiple=True,
    default=("deltas", "vertices"),
    show_default=True,
)
@_options.runner_options
@_options.plotting
def pairplots(figures_dir, n_components, features, overwrite, tasks, raise_on_error):
    """Generate PCA pairplots for baseline deltas and/or raw vertices."""
    from jacobs.core.euclidean import load_features
    from jacobs.core.pca.plot import PCAPairplotsRunner
    from jacobs.core.style import get_subject_colors

    figures_dir = figures_dir / f"pca_{n_components}"

    for kind in features:
        load = partial(load_features, subtract_baseline=kind == "deltas")

        runner = PCAPairplotsRunner(
            figures_dir,
            load,
            colors=get_subject_colors(),
            n_components=n_components,
            prefix=kind,
        )

        runner.run(
            overwrite=overwrite,
            tasks=tasks or None,
            raise_on_error=raise_on_error,
        )
