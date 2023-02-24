# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
from typing import Optional, Sequence, Union

import torch
from torch.optim import Optimizer
import torch.nn.functional as F

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)

__all__ = ['LinearizeConv']

class LinearConv(torch.nn.Module):
    """Convert Conv1x1s to Linear as the latter has a fused bias add."""

    def __init__(self, weight, bias):
        super().__init__()
        self.layer = torch.nn.Linear(weight.shape[0], weight.shape[1])
        self.layer.weight = torch.nn.parameter.Parameter(weight[:, :, 0, 0].contiguous())
        self.layer.bias = torch.nn.parameter.Parameter(bias)

    def forward(self, x):
        if x.shape[1] > self.layer.weight.shape[0]:
            return x[:, :self.layer.weight.shape[0], :, :]
        elif x.shape[1] * 2 == self.layer.weight.shape[0]:
            return torch.hstack([x, x])
        print('slow path', x.shape, self.layer.weight.shape)
        # Reshaping seems to trigger a different codepath with better performance? It's not clear...
        N, C_in, H, W = x.shape
        return self.layer(x.permute(0, 2, 3, 1).reshape(-1, C_in)).reshape(N, H, W, -1).permute(0, 3, 1, 2)


def apply_conv_linearization(model: torch.nn.Module,
                  optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None) -> torch.nn.Module:
    transforms = {
        torch.nn.Conv2d: functools.partial(
            _maybe_replace_conv2d,
        )
    }
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)

    return model


class LinearizeConv(Algorithm):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_LOAD

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        assert state.model is not None
        apply_conv_linearization(state.model, optimizers=state.optimizers)

def _maybe_replace_conv2d(
    module: torch.nn.Conv2d,
    module_index: int,
):
    if module.kernel_size == 1 or module.kernel_size == (1, 1):
        return LinearConv(module.weight, module.bias)
    return None
