"""Small terminal UI helpers for the interactive hardware setup."""

from __future__ import annotations

import os
import sys
import time
from typing import Callable


class Style:
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    cyan = "\033[36m"
    green = "\033[32m"
    yellow = "\033[33m"
    red = "\033[31m"
    magenta = "\033[35m"


def color(text: str, value: str) -> str:
    if not _use_color():
        return text
    return f"{value}{text}{Style.reset}"


def title(text: str) -> str:
    return color(text, Style.bold + Style.cyan)


def ok(text: str) -> str:
    return color(text, Style.green)


def warn(text: str) -> str:
    return color(text, Style.yellow)


def error(text: str) -> str:
    return color(text, Style.red)


def muted(text: str) -> str:
    return color(text, Style.dim)


def show_banner(print_fn: Callable[[str], None] = print) -> None:
    banner = [
        "███╗   ██╗ ██████╗ ███╗   ██╗██╗  ██╗██╗   ██╗███╗   ███╗ █████╗ ███╗   ██╗",
        "████╗  ██║██╔═══██╗████╗  ██║██║  ██║██║   ██║████╗ ████║██╔══██╗████╗  ██║",
        "██╔██╗ ██║██║   ██║██╔██╗ ██║███████║██║   ██║██╔████╔██║███████║██╔██╗ ██║",
        "██║╚██╗██║██║   ██║██║╚██╗██║██╔══██║██║   ██║██║╚██╔╝██║██╔══██║██║╚██╗██║",
        "██║ ╚████║╚██████╔╝██║ ╚████║██║  ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║██║ ╚████║",
        "╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝",
    ]
    for line in banner:
        print_fn(color(line, Style.bold + Style.magenta))
        if _animated():
            time.sleep(0.025)
    frames = ("□□□□□", "■□□□□", "■■□□□", "■■■□□", "■■■■□", "■■■■■")
    for frame in frames:
        if _animated():
            sys.stdout.write("\r" + color(f"booting setup {frame}", Style.cyan))
            sys.stdout.flush()
            time.sleep(0.08)
    if _animated():
        sys.stdout.write("\r" + " " * 32 + "\r")
        sys.stdout.flush()
    print_fn(color("Real teleop setup wizard", Style.bold + Style.cyan))
    print_fn(muted("Follow the prompts. Connect only the device requested at each step."))


def section(index: int, total: int, text: str, print_fn: Callable[[str], None] = print) -> None:
    print_fn("")
    print_fn(title(f"[{index}/{total}] {text}"))


def success(text: str, print_fn: Callable[[str], None] = print) -> None:
    print_fn(ok(f"✓ {text}"))


def info(text: str, print_fn: Callable[[str], None] = print) -> None:
    print_fn(color(f"• {text}", Style.cyan))


def _use_color() -> bool:
    return os.environ.get("NO_COLOR") is None and sys.stdout.isatty()


def _animated() -> bool:
    return _use_color() and os.environ.get("HANDUMI_NO_ANIMATION") != "1"
