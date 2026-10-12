import functools
from pathlib import Path

import click

figures_dir = click.option(
    "--figures-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="FIGURES_DIR",
    required=True,
)

tasks = click.option(
    "--task",
    "tasks",
    multiple=True,
    help="Task to run (repeatable). Defaults to all tasks.",
)

outputs_dir = click.option(
    "--outputs-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="OUTPUTS_DIR",
    required=True,
)


def plotting(func):
    """Apply jacobs matplotlib style before running the command."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from jacobs.core.style import update_mpl_params

        update_mpl_params()
        return func(*args, **kwargs)

    return wrapper
