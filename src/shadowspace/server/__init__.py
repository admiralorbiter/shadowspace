"""shadowspace.server — Flask workbench application.

Sprint 0: Minimal stub confirming the entry point and factory pattern.
Sprint 3b: Full workbench with PCA scatter, integrity panels, and saved-view atlas.
"""

from flask import Flask

from shadowspace.server.routes import workbench_bp


def create_app() -> Flask:
    """Flask application factory.

    Returns a configured Flask instance with the workbench blueprint registered.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(workbench_bp)
    return app
