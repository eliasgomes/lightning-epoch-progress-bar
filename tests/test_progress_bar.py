from unittest.mock import MagicMock

import pytest

from lightning_epoch_progress_bar import EpochCountColumn, EpochProgressBar


def _make_task(completed: float, total: float) -> MagicMock:
    task = MagicMock()
    task.completed = completed
    task.total = total
    return task


def _make_trainer(max_epochs: int | None = 10, current_epoch: int = 0) -> MagicMock:
    trainer = MagicMock()
    trainer.max_epochs = max_epochs
    trainer.current_epoch = current_epoch
    return trainer


def _started_bar(trainer: MagicMock) -> EpochProgressBar:
    bar = EpochProgressBar()
    bar._init_progress(trainer)
    bar.on_train_start(trainer, MagicMock())
    return bar


class TestEpochCountColumn:
    def test_mid_epoch(self) -> None:
        assert EpochCountColumn().render(_make_task(3.5, 100)).plain == "3/99"

    def test_epoch_end(self) -> None:
        assert EpochCountColumn().render(_make_task(4, 100)).plain == "4/99"

    def test_training_complete(self) -> None:
        assert EpochCountColumn().render(_make_task(100, 100)).plain == "99/99"


class TestEpochProgressBar:
    def test_init(self) -> None:
        bar = EpochProgressBar()
        assert bar._epoch_progress is None
        assert bar._epoch_task_id is None
        assert bar._leave is False

    @staticmethod
    def _make_metrics_trainer(**pbar_metrics: float) -> MagicMock:
        trainer = MagicMock()
        trainer.progress_bar_metrics = pbar_metrics
        trainer.state.fn = None
        trainer.loggers = [MagicMock(version=0)]
        return trainer

    def test_get_metrics_removes_v_num(self) -> None:
        trainer = self._make_metrics_trainer(val_loss=0.1)
        assert "v_num" not in EpochProgressBar().get_metrics(trainer, MagicMock())

    def test_get_metrics_preserves_other_keys(self) -> None:
        trainer = self._make_metrics_trainer(val_loss=0.5, ESR=0.01)
        metrics = EpochProgressBar().get_metrics(trainer, MagicMock())
        assert "val_loss" in metrics
        assert "ESR" in metrics

    def test_train_description_omits_total(self) -> None:
        assert EpochProgressBar()._get_train_description(5) == "Epoch 5"

    def test_on_train_start_creates_epoch_progress(self) -> None:
        bar = _started_bar(_make_trainer(max_epochs=100))
        assert bar._epoch_progress is not None
        assert bar._epoch_task_id is not None
        bar._stop_epoch_progress()

    def test_on_train_start_skips_when_max_epochs_is_none(self) -> None:
        bar = _started_bar(_make_trainer(max_epochs=None))
        assert bar._epoch_progress is None
        assert bar._epoch_task_id is None

    def test_on_train_epoch_end_updates_completed(self) -> None:
        trainer = _make_trainer(max_epochs=10, current_epoch=3)
        bar = _started_bar(trainer)
        bar.on_train_epoch_end(trainer, MagicMock())
        assert bar._epoch_progress is not None and bar._epoch_task_id is not None
        assert bar._epoch_progress.tasks[bar._epoch_task_id].completed == 4
        bar._stop_epoch_progress()

    def test_stop_epoch_progress_transient(self) -> None:
        bar = _started_bar(_make_trainer())
        assert bar._epoch_progress is not None
        assert bar._epoch_progress.live.is_started
        bar._stop_epoch_progress()
        assert not bar._epoch_progress.live.is_started
        assert bar._epoch_progress.live.transient

    def test_stop_epoch_progress_leave(self) -> None:
        bar = _started_bar(_make_trainer())
        bar._stop_epoch_progress(leave=True)
        assert bar._epoch_progress is not None
        assert not bar._epoch_progress.live.transient

    def test_stop_epoch_progress_noop_when_none(self) -> None:
        EpochProgressBar()._stop_epoch_progress()

    def test_on_train_end_leaves_epoch_bar(self) -> None:
        trainer = _make_trainer()
        bar = _started_bar(trainer)
        bar.on_train_end(trainer, MagicMock())
        assert bar._epoch_progress is not None
        assert not bar._epoch_progress.live.is_started
        assert not bar._epoch_progress.live.transient
        assert bar._progress_stopped

    def test_on_exception_stops_all(self) -> None:
        trainer = _make_trainer()
        bar = _started_bar(trainer)
        bar.on_exception(trainer, MagicMock(), RuntimeError("test"))
        assert bar._epoch_progress is not None
        assert not bar._epoch_progress.live.is_started
        assert bar._epoch_progress.live.transient
        assert bar._progress_stopped


if __name__ == "__main__":
    pytest.main()
