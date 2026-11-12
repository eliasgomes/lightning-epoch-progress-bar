"""Epoch-level Rich progress bar for PyTorch Lightning training."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class EpochCountColumn(ProgressColumn):
    """Shows the epoch count using Lightning's ``Epoch N/max-1`` convention."""

    def render(self, task: Task) -> Text:
        total = int(task.total) - 1 if task.total else 0
        current = min(int(task.completed), total)
        return Text(f"{current}/{total}")


class EpochProgressBar(RichProgressBar):
    """
    Rich progress bar with the current epoch's batches on top and overall epoch
    progress (with elapsed and remaining time) below.

    Both bars are transient, so the terminal is left clean on normal completion
    and on Ctrl+C. The epoch bar stays on screen once training finishes.
    """

    def __init__(self) -> None:
        super().__init__(leave=False)
        self._epoch_progress: Progress | None = None
        self._epoch_task_id: TaskID | None = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        if self.progress is not None:
            self.progress.live.transient = True
        if trainer.max_epochs is not None:
            self._epoch_progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    complete_style=self.theme.progress_bar,
                    finished_style=self.theme.progress_bar_finished,
                ),
                EpochCountColumn(),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                transient=True,
            )
            self._epoch_task_id = self._epoch_progress.add_task("Epochs", total=trainer.max_epochs)
            self._epoch_progress.start()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self._epoch_progress is not None and self._epoch_task_id is not None:
            fraction = (batch_idx + 1) / self.total_train_batches
            self._epoch_progress.update(
                self._epoch_task_id, completed=trainer.current_epoch + fraction
            )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self._epoch_progress is not None and self._epoch_task_id is not None:
            self._epoch_progress.update(self._epoch_task_id, completed=trainer.current_epoch + 1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._stop_progress()
        self._stop_epoch_progress(leave=True)

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._stop_progress()
        self._stop_epoch_progress()

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self._stop_progress()
        self._stop_epoch_progress()

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, int | str | float | dict[str, float]]:
        items = super().get_metrics(trainer, pl_module)
        # Always 0 with a single logger, so it's just noise.
        items.pop("v_num", None)
        return items

    def _get_train_description(self, current_epoch: int) -> str:
        # The total is redundant with the epoch bar below.
        return f"Epoch {current_epoch}"

    def _stop_epoch_progress(self, leave: bool = False) -> None:
        if self._epoch_progress is not None:
            if leave:
                self._epoch_progress.live.transient = False
            self._epoch_progress.stop()
