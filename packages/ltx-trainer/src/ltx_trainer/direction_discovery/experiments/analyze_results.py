"""Analyze and rank direction discovery results."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def analyze_results(
    metrics_path: str,
    output_path: Optional[str] = None,
    top_k: int = 10,
) -> None:
    """
    Analyze direction discovery results and generate ranking.

    Args:
        metrics_path: Path to metrics.json from experiment
        output_path: Optional path to save analysis
        top_k: Number of top directions to display
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        console.print(f"[red]Error: Metrics file not found: {metrics_path}[/red]")
        return

    # Load results
    with open(metrics_path) as f:
        data = json.load(f)

    results = data["results"]
    config = data["config"]

    console.print("[bold cyan]Direction Discovery Analysis[/bold cyan]")
    console.print()
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Method: {config['method']}")
    console.print(f"  Directions: {config['num_directions']}")
    console.print(f"  Metric: {config['metric']}")
    console.print()

    # Sort by quality score
    results_sorted = sorted(results, key=lambda r: r["quality_score"], reverse=True)

    # Create ranking table
    table = Table(title=f"Top {top_k} Directions by Quality Score")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Dir ID", style="magenta", width=8)
    table.add_column("Quality", style="green", width=12)
    table.add_column("Strength", style="yellow", width=12)
    table.add_column("Reversibility", style="red", width=14)

    for rank, result in enumerate(results_sorted[:top_k]):
        table.add_row(
            str(rank + 1),
            str(result["direction_id"]),
            f"{result['quality_score']:.6f}",
            f"{result['strength_score']:.6f}",
            f"{result['reversibility_score']:.8f}",
        )

    console.print(table)

    # Statistics
    console.print()
    console.print("[bold]Statistics:[/bold]")

    quality_scores = [r["quality_score"] for r in results]
    strength_scores = [r["strength_score"] for r in results]
    reversibility_scores = [r["reversibility_score"] for r in results]

    console.print(f"  Quality Score - Mean: {sum(quality_scores) / len(quality_scores):.6f}, "
                  f"Max: {max(quality_scores):.6f}, Min: {min(quality_scores):.6f}")
    console.print(f"  Strength Score - Mean: {sum(strength_scores) / len(strength_scores):.6f}, "
                  f"Max: {max(strength_scores):.6f}, Min: {min(strength_scores):.6f}")
    console.print(f"  Reversibility - Mean: {sum(reversibility_scores) / len(reversibility_scores):.8f}, "
                  f"Max: {max(reversibility_scores):.8f}, Min: {min(reversibility_scores):.8f}")

    # Save analysis
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        analysis = {
            "ranking": [
                {
                    "rank": rank + 1,
                    **result,
                }
                for rank, result in enumerate(results_sorted)
            ],
            "statistics": {
                "quality_score": {"mean": sum(quality_scores) / len(quality_scores), 
                                  "max": max(quality_scores), 
                                  "min": min(quality_scores)},
                "strength_score": {"mean": sum(strength_scores) / len(strength_scores), 
                                   "max": max(strength_scores), 
                                   "min": min(strength_scores)},
                "reversibility_score": {"mean": sum(reversibility_scores) / len(reversibility_scores), 
                                        "max": max(reversibility_scores), 
                                        "min": min(reversibility_scores)},
            },
        }

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        console.print(f"\n[green]✓[/green] Analysis saved to: {output_path}")


if __name__ == "__main__":
    typer.run(analyze_results)