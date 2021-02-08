from pathlib import Path
from typing import Any

from py9lib.io_ import read_yaml, write_yaml
from py9lib.log import get_logger

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FN = PROJECT_ROOT / "autorebalance.yaml"


LOG = get_logger("autorebalance")


def secret_fn(basename: str) -> Path:
    return PROJECT_ROOT.joinpath("secrets").joinpath(basename).absolute()


def data_fn(basename: str) -> Path:
    return PROJECT_ROOT.joinpath("data").joinpath(basename).absolute()


def config() -> dict[str, Any]:
    return read_yaml(CONFIG_FN)


class ConfigWriteback:
    def __init__(self) -> None:
        self.conf = read_yaml(CONFIG_FN)

    def __enter__(self) -> dict[str, Any]:
        return self.conf

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        if exc_type is None:
            write_yaml(CONFIG_FN, self.conf, indent=4)
