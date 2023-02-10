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

try:
    from xformers.triton.layer_norm import FusedLayerNorm as _FusedLayerNorm
    is_xformers_installed = True
except:
    print('Warning: xformers is not installed.')
    is_xformers_installed = False


log = logging.getLogger(__name__)

__all__ = ['FusedLayerNorm']


def apply_fusedlayernorm(model: torch.nn.Module,
                  optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None) -> torch.nn.Module:
    transforms = {
        torch.nn.LayerNorm: functools.partial(
            _replace_layer_norm,
        ),
    }
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)

    return model


class FusedLayerNorm(Algorithm):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_LOAD

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        assert state.model is not None
        apply_fusedlayernorm(state.model, optimizers=state.optimizers)

def _replace_layer_norm(
    module: torch.nn.LayerNorm,
    module_index: int,
):
    if not is_xformers_installed:
        return None
    layer = _FusedLayerNorm(module.normalized_shape)
    layer.weight = module.weight
    layer.bias = module.bias
    return layer
