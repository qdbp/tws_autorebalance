from configparser import ConfigParser
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FN = PROJECT_ROOT.joinpath("autorebalance.ini").__str__()


def secret_fn(basename: str) -> Path:
    return PROJECT_ROOT.joinpath("secrets").joinpath(basename).absolute()


def data_fn(basename: str) -> Path:
    return PROJECT_ROOT.joinpath("data").joinpath(basename).absolute()


def config() -> ConfigParser:
    key = "__autorebalance_config"
    # this caching is done to ensure that config changes do not take effect
    # until the application is restarted
    if key not in globals():
        cp = ConfigParser()
        cp.read(CONFIG_FN)
        globals()[key] = cp
    return cast(ConfigParser, globals()[key])
