import click


@click.group()
def _cli():
    pass


@_cli.command()
def generate_data():
    """
    Generates data for MonsterMatch and prints the C# file to standard out.

    This is used by an non-negative matrix factorization algorithm to make recommendations. It is the baseline data.
    """
    from ..data import generate_array_data_file
    generate_array_data_file()


def main():
    _cli()
