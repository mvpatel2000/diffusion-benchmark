# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
from typing import Optional, Sequence, Union

import torch
from torch.optim import Optimizer

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as APEXFusedLayerNorm
    APEX_INSTALLED = True
except ImportError as e:
    APEX_INSTALLED = False

log = logging.getLogger(__name__)

__all__ = ['FusedGroupNorm']

def check_if_apex_installed():
    if not APEX_INSTALLED:
        raise ImportError(
            'https://github.com/NVIDIA/apex is not installed. The Fused LayerNorm algorithm cannot be applied. The MosaicML Docker Images (https://hub.docker.com/r/mosaicml/pytorch) contain a copy of APEX for easy use.'
        )

class LayerGroupNorm(torch.nn.Module):
    def __init__(self, weight, bias, G, C):
        super().__init__()
        self.vars_set = False
        self.weight = weight
        self.bias = bias
        self.G = G
        self.C = C

    def forward(self, x):
        N, _, H, W = x.shape
        if not self.vars_set:
            self.layer = torch.nn.LayerNorm([self.C//self.G, H, W], elementwise_affine=False)
            self.weight = torch.nn.parameter.Parameter(self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(1, self.C, H, W))
            self.bias = torch.nn.parameter.Parameter(self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(1, self.C, H, W))
            self.vars_set = True
            # TODO: How to reshape weight / bias into FLN compatible way?
            if self.G > 1:
                raise ValueError('More than 1 group is not supported yet.')

        out = self.layer(x.reshape(-1, self.C//self.G, H, W)).reshape(-1, self.C, H, W)
        out = out * self.weight.expand(N, -1, -1, -1) + self.bias.expand(N, -1, -1, -1)
        return out


def apply_groupnorm(
    model: torch.nn.Module,
    optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,
) -> torch.nn.Module:
    check_if_apex_installed()

    transforms = {
        torch.nn.GroupNorm: functools.partial(
            _replace_group_norm,
        )
    }
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)

    return model


class FusedGroupNorm(Algorithm):
    def __init__(self):
        super().__init__()

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
    return LayerGroupNorm(module.weight, module.bias, module.num_groups, module.num_channels)
