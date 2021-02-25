#!/usr/bin/env python
import click

from src.app import arb_entrypoint


@click.command()
@click.option("--debug", type=bool, default=False, is_flag=True)
@click.option("--disarm", type=bool, default=False, is_flag=True)
def main(debug: bool, disarm: bool) -> None:
    arb_entrypoint(debug=debug, disarm=disarm)


if __name__ == "__main__":
    main()
