import click

from .bcv import bcv
from .pca import pca


@click.group()
def euclidean():
    """Analyses using Euclidean."""


euclidean.add_command(pca)
euclidean.add_command(bcv)
