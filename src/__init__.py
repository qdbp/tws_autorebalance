from configparser import ConfigParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def secret_fn(basename: str) -> Path:
    return PROJECT_ROOT.joinpath("secrets").joinpath(basename).absolute()


def data_fn(basename: str) -> Path:
    return PROJECT_ROOT.joinpath("data").joinpath(basename).absolute()


def config() -> ConfigParser:
    key = "__autorebalance_config"
    # this caching is done to ensure that config-changes do not take effect until the
    # application is restarted
    if key not in globals():
        cp = ConfigParser()
        cp.read(PROJECT_ROOT.joinpath("autorebalance.ini").__str__())
        globals()[key] = cp
    return globals()[key]
