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

__all__ = ['FusedGroupNorm']

class GroupNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def apply_groupnorm(model: torch.nn.Module,
                  optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None) -> torch.nn.Module:
    transforms = {
        torch.nn.GroupNorm: functools.partial(
            _replace_group_norm,
        )
    }
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)

    return model


class FusedGroupNorm(Algorithm):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_LOAD

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        assert state.model is not None
        apply_groupnorm(state.model, optimizers=state.optimizers)

def _replace_group_norm(
    module: torch.nn.GroupNorm,
    module_index: int,
):
    return GroupNorm()
