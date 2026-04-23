from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    from .app import main as app_main

    app_main(argv)


__all__ = ["main"]
