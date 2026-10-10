import click

from .enigma import enigma
from .euclidean import euclidean
from .fast import fast
from .lddmm import lddmm


@click.group()
def cli():
    """Jacobs dataset analyses."""


cli.add_command(enigma)
cli.add_command(fast)
cli.add_command(lddmm)
cli.add_command(euclidean)
