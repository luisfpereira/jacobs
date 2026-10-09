import click

from .enigma import enigma
from .fast import fast


@click.group()
def cli():
    """Jacobs dataset analyses."""


cli.add_command(enigma)
cli.add_command(fast)
