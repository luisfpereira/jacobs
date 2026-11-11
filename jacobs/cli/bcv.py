"""Bi-cross-validation figures combining the Euclidean and LDDMM pipelines.

After both pipelines' ``evaluation-figures``, with ``EUCLIDEAN_OUTPUTS_DIR``
and ``LDDMM_OUTPUTS_DIR`` set (e.g. ``source config/local/bcv.sh``)::

    jacobs bcv panel
"""

from pathlib import Path

import click


@click.group()
def bcv():
    """Bi-cross-validation figures combining pipelines."""


@bcv.command()
@click.option(
    "--euclidean-outputs-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="EUCLIDEAN_OUTPUTS_DIR",
    required=True,
)
@click.option(
    "--lddmm-outputs-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="LDDMM_OUTPUTS_DIR",
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Defaults to an artifacts directory next to the Euclidean outputs.",
)
def panel(euclidean_outputs_dir, lddmm_outputs_dir, output_dir):
    """Euclidean and geometric reconstruction errors over scaling factors."""
    from jacobs.core.bcv import (
        get_evaluation_figures_dir,
        make_errors_scaling_panel,
    )

    if output_dir is None:
        output_dir = euclidean_outputs_dir.parent / "artifacts"

    output_dir.mkdir(parents=True, exist_ok=True)

    make_errors_scaling_panel(
        {
            "Euclidean": get_evaluation_figures_dir(
                euclidean_outputs_dir / "artifacts"
            ),
            "Geometric": get_evaluation_figures_dir(lddmm_outputs_dir / "artifacts"),
        },
        output_dir / "bicv-rank-selection.pdf",
    )
