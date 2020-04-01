from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def secret_fn(basename: str) -> str:
    return PROJECT_ROOT.joinpath("secrets").joinpath(basename).absolute().__str__()
