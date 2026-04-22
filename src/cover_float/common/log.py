from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import time
from collections.abc import Generator, Iterable, Sized
from contextlib import contextmanager
from queue import Queue
from types import TracebackType
from typing import Any, Callable, TypeVar

from rich import console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
)

STATUS_LEVEL_NUM = 25
logging.addLevelName(STATUS_LEVEL_NUM, "STATUS")


class ModelLogger(logging.Logger):
    task_id: TaskID
    msg_queue: Queue[Any]

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def status(self, message: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if self.isEnabledFor(STATUS_LEVEL_NUM):
            self._log(STATUS_LEVEL_NUM, message, args, **kwargs)

    @contextmanager
    def progress_bar(
        self, status: str | None = None, *, total: float | None = 100, show_m_of_n: bool = False
    ) -> Generator[BarHandle, None, None]:
        handle = BarHandle(self.msg_queue, self.task_id)

        if status is not None:
            handle.update(total=total, completed=0, status=status, m_of_n=show_m_of_n)
        else:
            handle.update(total=total, completed=0, m_of_n=show_m_of_n)

        try:
            yield handle
        finally:
            handle.update(total=None, completed=0, m_of_n=False)


logging.setLoggerClass(ModelLogger)


class ExcludeStatusFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != STATUS_LEVEL_NUM


class OnlyStatusFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == 25


T = TypeVar("T")


class BarHandle:
    def __init__(self, queue: Queue[Any], task_id: TaskID) -> None:
        self._queue = queue
        self._task_id = task_id

    # Match Progress.advance exactly
    def advance(self, advance: float = 1) -> None:
        # self._progress.advance(self._task_id, advance)
        self._queue.put({"action": "advance", "args": [self._task_id, advance], "kwargs": {}})

    # Match Progress.update exactly — forward all kwargs through
    def update(
        self,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,  # noqa: ANN401
    ) -> None:
        self._queue.put(
            {
                "action": "update",
                "args": [self._task_id],
                "kwargs": {
                    "total": total,
                    "completed": completed,
                    "advance": advance,
                    "description": description,
                    "visible": visible,
                    "refresh": refresh,
                    **fields,
                },
            }
        )

    def track(
        self,
        sequence: Iterable[T],
        *,
        total: float | None = None,
        completed: int = 0,
        update_period: float = 0.1,
    ) -> Iterable[T]:
        if total is None and isinstance(sequence, Sized):
            total = len(sequence)

        last_update = time.monotonic() - update_period

        for x in sequence:
            yield x

            now = time.monotonic()
            completed += 1
            if now - last_update >= update_period:
                self.update(total=total, completed=completed)
                last_update = now


class OptionalColumn(ProgressColumn):
    def __init__(self, condition: Callable[[Task], bool], real_col: ProgressColumn) -> None:
        super().__init__()
        self._real_col = real_col
        self._condition = condition

    def render(self, task: Task) -> console.RenderableType:
        if not self._condition(task):
            return ""
        return self._real_col.render(task)


class LoggingHandler(logging.Handler):
    def __init__(self, reporter: StatusReporter, model_name: str) -> None:
        super().__init__()
        self.reporter = reporter
        self.model_name = model_name

    def emit(self, record: logging.LogRecord) -> None:
        self.reporter.status_update(self.model_name, record.msg)


class AsyncLoggingHandler(logging.handlers.QueueListener):
    def __init__(self, reporter: StatusReporter, listen_to: Queue[Any], *handlers: logging.Handler) -> None:
        super().__init__(listen_to, *handlers)

        self.reporter = reporter

    def handle_progress_update(self, record: dict[Any, Any]) -> None:
        try:
            action = record["action"]
            if action == "update":
                self.reporter.progress.update(*record["args"], **record["kwargs"])
            elif action == "advance":
                self.reporter.progress.advance(*record["args"], **record["kwargs"])
            elif action == "remove_task":
                model_name = record["args"][0]
                self.reporter.progress.remove_task(self.reporter.active_status_bars[model_name])
                self.reporter.active_status_bars.pop(model_name)
            else:
                logging.info(f"Failed to Log {record}")
        except Exception as e:
            logging.info(f"Failed to Log {record}", exc_info=e)

    def handle(self, record: logging.LogRecord | dict[Any, Any]) -> None:
        if isinstance(record, logging.LogRecord):
            super().handle(record)
        else:
            self.handle_progress_update(record)


class ProgressAwareLogHandler(logging.Handler):
    def __init__(self, progress: Progress) -> None:
        super().__init__()
        self.progress = progress
        # We use the rich handler as a formatter
        self.rich_handler = RichHandler(show_time=True, markup=True, console=progress.console)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.rich_handler.render_message(record, self.rich_handler.format(record))
            renderable = self.rich_handler.render(
                record=record,
                traceback=None,
                message_renderable=message,
            )
            self.progress.log(renderable)
        except Exception:
            self.handleError(record)


class StatusReporter:
    def __init__(self) -> None:
        self.active_status_bars: dict[str, TaskID] = {}
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/]"),
            TextColumn("[yellow]{task.fields[status]}[/]"),
            OptionalColumn(lambda t: t.total is not None, BarColumn()),
            OptionalColumn(lambda t: t.fields.get("m_of_n", False), MofNCompleteColumn()),
            OptionalColumn(lambda t: t.total is not None, TextColumn("{task.percentage:>3.0f}%")),
        )

        self.manager = multiprocessing.Manager()
        self.logging_queue = self.manager.Queue()

        # This is the actual handler that prints to console
        self.rich_handler = ProgressAwareLogHandler(self.progress)
        self.queue_listener = AsyncLoggingHandler(self, self.logging_queue, self.rich_handler)
        logging.getLogger().addHandler(logging.handlers.QueueHandler(self.logging_queue))

    def __enter__(self) -> StatusReporter:
        self.progress.start()
        self.queue_listener.start()

        return self

    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: TracebackType | None) -> None:
        self.queue_listener.stop()
        self.progress.stop()

    def start_model(self, model_name: str) -> TaskID:
        task_id = self.progress.add_task(f"{model_name}: ", status="Test Generation", total=None)
        self.active_status_bars[model_name] = task_id
        return task_id

    def stop_model(self, model_name: str) -> None:
        # This can be a race condition, if there is logging still in the queue
        # so we must enqueue the destruction
        self.logging_queue.put({"action": "remove_task", "args": [model_name], "kwargs": {}})

    def status_update(self, model_name: str, status_message: str) -> None:
        task = self.active_status_bars[model_name]
        self.progress.update(task, status=status_message)
