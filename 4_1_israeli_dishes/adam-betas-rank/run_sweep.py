"""Prepare or run the adam-betas-rank Israeli dishes experiment."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from optimizer_sweeps.cli import main

if __name__ == "__main__":
    main("adam-betas-rank")
