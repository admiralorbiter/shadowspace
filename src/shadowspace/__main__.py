"""Module entrypoint for python -m shadowspace."""

import sys

from shadowspace.cli import main

if __name__ == "__main__":
    sys.exit(main())
