#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")

import click

from data_prep_scripts.jlaunch_backends import backends_list


@click.group()
def cli():
    pass


for backend in backends_list:
    cli.add_command(backend)

if __name__ == "__main__":
    cli()
