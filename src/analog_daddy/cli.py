"""Console script for analog_daddy."""

from cyclopts import App
from rich.console import Console

app = App()
console = Console()


@app.default
def main() -> None:
    """Console script for analog_daddy."""
    console.print(
        "Replace this message by putting your code into analog_daddy.cli.main"
    )
    console.print(
        "See Cyclopts documentation at https://cyclopts.readthedocs.io/"
    )


if __name__ == "__main__":
    app()
