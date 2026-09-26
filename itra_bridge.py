"""Receive ITRA race captures from the companion browser extension."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components


_bridge = components.declare_component(
    "itra_browser_bridge",
    path=str(Path(__file__).resolve().parent / "itra_bridge_frontend"),
)


def receive_browser_capture(command: dict | None = None) -> dict | None:
    """Exchange extension captures and local-storage commands with the browser."""
    return _bridge(
        command=command or {},
        default=None,
        key="itra_browser_bridge",
    )
