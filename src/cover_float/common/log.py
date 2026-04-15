from __future__ import annotations

import io
import os
from collections.abc import Iterable
from typing import Any, TypeVar, overload

import tqdm


class LoggingContext(io.TextIOBase):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.error_count = 0
        self.last_print = ""

        self.write("")

    def write(self, s: str) -> int:
        processed_string = s.strip("\r").replace("\n", "\r\033[K")
        print("\033[K" + self.prefix + processed_string, end="\r")
        self.last_print = s

        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def increase_error_count(self) -> None:
        self.error_count += 1

    def set_prefix(self, prefix: str) -> None:
        self.prefix = prefix
        self.write("")

    def print_error(self, error: str) -> None:
        print(self.prefix + error)
        self.write(self.last_print)


_the_logging_context = LoggingContext("")


def log_info(s: str) -> None:
    _the_logging_context.write(s)


def log_error(e: str) -> None:
    _the_logging_context.print_error(e)
    _the_logging_context.increase_error_count()


T = TypeVar("T")


@overload
def progress_bar(maybe_iterable: Iterable[T], *args: Any, **kwargs: Any) -> tqdm.tqdm[T]: ...  # noqa: ANN401


@overload
def progress_bar(maybe_iterable: None = None, *args: Any, **kwargs: Any) -> tqdm.tqdm[object]: ...  # noqa: ANN401


def progress_bar(maybe_iterable: Iterable[T] | None = None, *args: Any, **kwargs: Any) -> tqdm.tqdm[T]:
    tqdm_arguments: dict[str, Any] = {
        "file": _the_logging_context,
        "ascii": False,
    }

    try:
        tqdm_arguments["ncols"] = (os.get_terminal_size().columns - len(_the_logging_context.prefix),)
    except OSError:
        # ncols is not available in CI or certain environments
        tqdm_arguments["ncols"] = None

    # This avoids any conflicts, and gives kwargs precedence
    real_kwargs = {**tqdm_arguments, **kwargs}

    if maybe_iterable is not None:
        return tqdm.tqdm(maybe_iterable, *args, **real_kwargs)
    else:
        return tqdm.tqdm(*args, **real_kwargs)


def set_prefix(s: str) -> None:
    _the_logging_context.set_prefix(s)
