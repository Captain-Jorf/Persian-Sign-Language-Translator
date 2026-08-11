from __future__ import annotations

import sys

from psl_translator.cli import main as cli_main


def main() -> int:
    # Double-clicking a CLI exe on Windows opens cmd, exits, and looks broken.
    # With no args we launch the small Tk GUI; with args we keep the CLI behavior.
    if len(sys.argv) == 1:
        from psl_translator.gui import main as gui_main

        return gui_main()
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())