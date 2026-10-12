import click

from jacobs.core.logging import configure_logging

from .enigma import enigma
from .euclidean import euclidean
from .fast import fast
from .lddmm import lddmm


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        case_sensitive=False,
    ),
    default="INFO",
    envvar="JACOBS_LOG_LEVEL",
    show_default=True,
    show_envvar=True,
)
def cli(log_level):
    """Jacobs dataset analyses."""
    configure_logging(log_level.upper())


cli.add_command(enigma)
cli.add_command(fast)
cli.add_command(euclidean)
cli.add_command(lddmm)
