import getpass
import socket
from pathlib import Path


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
