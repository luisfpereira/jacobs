import getpass
import socket
from pathlib import Path

from polpo.neuroimaging.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)


def in_frank():
    return socket.gethostname() == "frank"


def get_frank_user_scratch():
    # TODO: check use
    return Path(f"/scratch/{getpass.getuser()}")


def get_data_dir():
    return (
        Path("/scratch/data/maternal") if in_frank() else Path.home() / "data/maternal"
    )


def resolve_data_dir(data_dir=None):
    if data_dir is None:
        return get_data_dir()

    return data_dir


def find_experiment_dirs(outputs_dir, long_name=False, interleave=True):
    """Find subcortical experiment directories.

    Parameters
    ----------
    outputs_dir : path-like
        Directory containing structure-specific experiment directories.
    long_name : bool
        Whether to use long structure names as dictionary keys.
    interleave : bool
        Whether to order left and right hemisphere structures in an
        interleaved sequence.

    Returns
    -------
    experiment_dirs : dict
        Mapping from structure names to experiment directories, ordered
        according to the requested structure ordering.
    """
    all_structs = get_all_subcortical_structs(interleave=interleave)
    order = {struct: i for i, struct in enumerate(all_structs)}

    dirs = sorted(
        (
            path
            for path in outputs_dir.iterdir()
            if path.is_dir() and path.name in order
        ),
        key=lambda path: order[path.name],
    )

    structs = (
        [get_subcortical_struct_long_name(dir_.name) for dir_ in dirs]
        if long_name
        else [dir_.name for dir_ in dirs]
    )

    return dict(zip(structs, dirs))
