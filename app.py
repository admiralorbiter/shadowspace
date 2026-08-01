"""Shadowspace workbench entry point.

Run with:
    python app.py

Sprint 0:  Minimal Flask stub — confirms the entry point works.
Sprint 3b: Full workbench with dtour scatter, integrity panels, and saved-view atlas.
"""

from shadowspace.server import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
