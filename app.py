#!/usr/bin/env python3
"""
app.py - Hugging Face Spaces entrypoint for the fort_calc Gradio demo.

Spaces expects a top-level variable referencing the Gradio app. This file
imports the builder from src/gradio_ui.py and exposes it as `demo` so the
platform can serve it. Do NOT call demo.launch() here.
"""

from src.gradio_ui import _build_ui

# Build and expose the Gradio demo for Hugging Face Spaces
demo = _build_ui()
