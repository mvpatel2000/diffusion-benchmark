# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['PredictSpeedMonitor']


class PredictSpeedMonitor(Callback):
    """Logs predict throughput.

    The predict throughput in terms of number of samples per second is logged on the
    :attr:`.Event.PREDICT_BATCH_END` event if we have reached the ``window_size`` threshold.

    The wall clock train time is logged on every :attr:`.Event.PREDICT_BATCH_END` event.

    The average throughout over an epoch is logged on the :attr:`.Event.PREDICT_END` event.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import SpeedMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[PredictSpeedMonitor(window_size=100)],
            ... )

    The training throughput is logged by the :class:`.Logger` to the following keys as
    described below.

    +---------------------------------------+-------------------------------------------------------------+
    | Key                                   | Logged data                                                 |
    +=======================================+=============================================================+
    |                                       | Rolling average (over ``window_size`` most recent           |
    | ``throughput/predict_samples_per_sec` | batches) of the number of samples processed per second      |
    |                                       |                                                             |
    +---------------------------------------+-------------------------------------------------------------+
    | ``wall_clock/predict``                | Total elapsed predict time                                  |
    +---------------------------------------+-------------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
    """

    def __init__(self, window_size: int = 100):
        # Track the batch num samples and wct to compute throughput over a window of batches
        self.batch_start_num_samples = 0
        self.batch_start_wct = 0.0
        self.batch_wct_buffer: Deque[float] = deque(maxlen=window_size)
        self.batch_num_samples_buffer: Deque[int] = deque(maxlen=window_size)
        self.window_size = window_size

    def state_dict(self) -> Dict[str, Any]:
        return {
            'batch_start_num_samples': self.batch_start_num_samples,
            'batch_start_wct': self.batch_start_wct,
            'batch_wct_buffer': self.batch_wct_buffer,
            'batch_num_samples_buffer': self.batch_num_samples_buffer,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.batch_start_num_samples = state['batch_start_num_samples']
        self.batch_start_wct = state['batch_start_wct']
        self.batch_wct_buffer = deque(
            [x for x in state['batch_wct_buffer']],
            maxlen=self.window_size,
        )
        self.batch_num_samples_buffer = deque(
            [x for x in state['batch_num_samples_buffer']],
            maxlen=self.window_size,
        )

    def predict_batch_end(self, state: State, logger: Logger):
        batch_num_samples = int(state.predict_timestamp.sample) - self.batch_start_num_samples
        batch_wct = state.predict_timestamp.total_wct.total_seconds() - self.batch_start_wct

        # Add the new element
        self.batch_wct_buffer.append(batch_wct)
        self.batch_num_samples_buffer.append(batch_num_samples)

        # Log the throughput
        if len(self.batch_num_samples_buffer) == self.window_size:
            throughput = sum(self.batch_num_samples_buffer) / sum(self.batch_wct_buffer)
            logger.log_metrics({'throughput/predict_samples_per_sec': throughput}, step=state.predict_timestamp.batch.value)

        # Log the time
        logger.log_metrics({
            'wall_clock/predict': state.predict_timestamp.total_wct.total_seconds(),
        }, step=state.predict_timestamp.batch.value)

        # Reset the start time and num samples
        self.batch_start_wct = state.predict_timestamp.total_wct.total_seconds()
        self.batch_start_num_samples = int(state.predict_timestamp.sample)
