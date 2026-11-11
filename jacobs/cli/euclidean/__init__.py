import click

from .bcv import bcv
from .pca import pca
from .reference import reference


@click.group()
def euclidean():
    """Analyses using Euclidean."""


euclidean.add_command(pca)
euclidean.add_command(bcv)
euclidean.add_command(reference)
