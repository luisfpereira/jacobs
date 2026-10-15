import click

from .bcv import bcv
from .pca import pca
from .validation import validation


@click.group()
def lddmm():
    """Analyses using LDDMM."""


lddmm.add_command(validation)
lddmm.add_command(pca)
lddmm.add_command(bcv)
