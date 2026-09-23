"""Shared pieces for the MaleCNS mushroom body experiments.

The scripts in scripts/ and experiments/ are thin wrappers; everything they have
in common lives here so the randomisation controls and the coding/learning rules
are defined exactly once.
"""
from . import capacity, coding, data, door, odors, wiring

__all__ = ["capacity", "coding", "data", "door", "odors", "wiring"]
