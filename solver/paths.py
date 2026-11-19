"""Stable path to the project root (works from package modules and CLI)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
