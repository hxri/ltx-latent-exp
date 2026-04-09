#!/usr/bin/env python3
"""
Standalone demo script for direction discovery.
Run this to test the complete pipeline with a sample video.
"""

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltx_trainer.direction_discovery.experiments.run_discovery import run_direction_discovery

def main():
    """
    Demo entry point.
    
    Usage:
        python scripts/direction_discovery_demo.py \\
            --video video.mp4 \\
            --checkpoint-path /path/to/ltx2.safetensors \\
            --method random \\
            --num-directions 5
    """
    typer.run(run_direction_discovery)

if __name__ == "__main__":
    main()