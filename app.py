#!/usr/bin/env python3
"""
app.py - Hugging Face Spaces entrypoint for the fort_calc Gradio demo.

This file builds the Gradio demo and launches it immediately. Spaces and the
Gradio example in the docs sometimes call demo.launch() directly from app.py.
"""

from src.gradio_ui import _build_ui

# Build and launch the Gradio demo immediately on import
demo = _build_ui()
demo.launch()
