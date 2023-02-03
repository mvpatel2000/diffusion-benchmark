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

__all__ = ['Conv1x1']

class LinearConv(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight[:, :, 0, 0].contiguous()
        self.bias = bias

    def forward(self, x):
        return F.linear(x.permute(0, 2, 3, 1), self.weight, self.bias).permute(0, 3, 1, 2)


def apply_conv1x1(model: torch.nn.Module,
                  optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None) -> torch.nn.Module:
    transforms = {}
    transforms = {
        torch.nn.Conv2d: functools.partial(
            _maybe_replace_conv2d,
        )
    }
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)

    return model


class Conv1x1(Algorithm):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_LOAD

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        assert state.model is not None
        state.model.to(memory_format=torch.channels_last)
        apply_conv1x1(state.model, optimizers=state.optimizers)

def _maybe_replace_conv2d(
    module: torch.nn.Conv2d,
):
    if module.kernel_size == 1 or module.kernel_size == (1, 1):
        return LinearConv(module.weight, module.bias)
    return None
