"""Reproducible comparisons with explicit task and uncertainty capabilities.

Optional neural dependencies are imported only by their own adapters.
"""
from .contracts import Panel, Forecast, CapabilityError, select_panel
from .protocol import BenchmarkProtocol, profile
from .registry import fit_model, load_model, save_model, model_registry

__all__=["Panel", "Forecast", "CapabilityError", "select_panel", "BenchmarkProtocol", "profile",
         "fit_model", "load_model", "save_model", "model_registry"]
