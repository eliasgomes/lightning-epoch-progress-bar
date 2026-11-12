# lightning-epoch-progress-bar

A [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) callback that adds an epoch-level
[Rich](https://github.com/Textualize/rich) progress bar under Lightning's usual per-batch bar, so you
can see how far along the whole run is, how long it has taken and roughly how long is left.

![Per-batch bar on top, epoch progress bar with elapsed and remaining time below](https://github.com/user-attachments/assets/56e00ad7-04dd-451f-aa44-563e7e8e643b)

- The epoch bar moves forward with every batch, not only at the end of each epoch, and shows elapsed
  and remaining time.
- The epoch count uses Lightning's own `N/max_epochs-1` convention.
- Both bars are transient, so a run stopped with Ctrl+C leaves the terminal clean. When training
  finishes normally, the epoch bar stays on screen.
- The per-batch bar shows just `Epoch N`, since the total is already on the epoch bar, and it hides
  `v_num`, which is always `0` with a single logger.

## Install

```console
pip install git+https://github.com/eliasgomes/lightning-epoch-progress-bar
```

It needs `pytorch_lightning` 2.x and `rich`, which you'll already have if you use Lightning's
`RichProgressBar`.

## Usage

Pass it to the trainer like any other callback:

```python
import pytorch_lightning as pl

from lightning_epoch_progress_bar import EpochProgressBar

trainer = pl.Trainer(max_epochs=100, callbacks=[EpochProgressBar()])
```

The epoch bar only appears when `max_epochs` is set. Like any progress bar callback, it can't be
combined with `enable_progress_bar=False`.

> [!NOTE]
> It subclasses `pytorch_lightning.callbacks.RichProgressBar`, so it works with
> `pytorch_lightning.Trainer`. The unified `lightning` package's `lightning.pytorch.Trainer` uses a
> separate class hierarchy and won't accept it.

### With Neural Amp Modeler

[Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)'s full trainer (`nam-full`)
can load callbacks listed in `learning.json`. Install this package into your NAM environment, then
add it under `"trainer"`:

```json
"trainer": {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 100,
    "callbacks": [
        {
            "name": "lightning_epoch_progress_bar.EpochProgressBar",
            "args": [],
            "kwargs": {}
        }
    ]
}
```

## Development

```console
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```

## License

[MIT](LICENSE)
