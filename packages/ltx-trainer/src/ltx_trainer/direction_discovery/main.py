"""Main entry point for direction discovery experiments."""

import typer

from ltx_trainer.direction_discovery.experiments.analyze_results import analyze_results
from ltx_trainer.direction_discovery.experiments.cycle_consistency import run_cycle_consistency
from ltx_trainer.direction_discovery.experiments.run_discovery import run_direction_discovery
from ltx_trainer.direction_discovery.experiments.transfer_saved_direction import run_transfer_saved_direction

app = typer.Typer(no_args_is_help=True)

app.command()(run_direction_discovery)
app.command()(analyze_results)
app.command()(run_cycle_consistency)
app.command()(run_transfer_saved_direction)

if __name__ == "__main__":
    app()