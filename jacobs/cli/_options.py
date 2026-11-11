import functools
from pathlib import Path

import click


def compose(*decorators):
    """Combine decorators into one, applied in the listed order."""

    def decorate(func):
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return decorate


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

raise_on_error = click.option(
    "--raise/--no-raise",
    "raise_on_error",
    default=True,
    show_default=True,
    help="Stop at the first failed task and re-raise its error. "
    "With --no-raise, record the failure and continue.",
)

overwrite = click.option(
    "--overwrite",
    is_flag=True,
    help="Rerun tasks already marked as completed.",
)


runner_options = compose(overwrite, tasks, raise_on_error)


def plotting(func):
    """Apply jacobs matplotlib style before running the command."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from jacobs.core.style import update_mpl_params

        update_mpl_params()
        return func(*args, **kwargs)

    return wrapper


features = click.option(
    "--features",
    type=click.Choice(["deltas", "vertices"]),
    default="deltas",
    show_default=True,
    help="Represent meshes as baseline deltas or registered vertices. "
    "Use the same choice for every step, and one outputs directory per choice.",
)

noise_floor_dir = click.option(
    "--noise-floor-dir",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="NOISE_FLOOR_DIR",
    default=None,
    help="Outputs directory of `jacobs enigma noise-floor`. If given, error "
    "figures mark each structure's noise floor.",
)
