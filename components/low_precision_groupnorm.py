# Copyright 2022 MosaicML Agent authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch
import torch.nn.functional as F

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, Precision, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def _cast_if_autocast_enabled(hidden_states):
    if not torch.is_autocast_enabled():
        return hidden_states
    else:
        return torch.cuda.amp.autocast_mode._cast(hidden_states, torch.get_autocast_gpu_dtype())


class LPGroupNorm(torch.nn.GroupNorm):

    def __init__(self, layer):
        super().__init__(
            num_groups=layer.num_groups,
            num_channels=layer.num_channels,
            eps=layer.eps,
            affine=layer.affine,
        )

        with torch.no_grad():
            self.weight.copy_(layer.weight)
            self.bias.copy_(layer.bias)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight)
        downcast_bias = _cast_if_autocast_enabled(self.bias)
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.group_norm(downcast_x, self.num_groups, downcast_weight, downcast_bias, self.eps)


def to_LPGroupNorm(layer: torch.nn.Module, module_index: int) -> LPGroupNorm:
    assert isinstance(layer,
                      torch.nn.GroupNorm), 'The replacement policy will look for all instances of torch.nn.GroupNorm'
    return LPGroupNorm(layer)



def apply_low_precision_groupnorm(model, optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                                  precision: Precision):

    if (precision != Precision.AMP_FP16 and precision != Precision.AMP_BF16):
        warnings.warn(NoEffectWarning('Low Precision GroupNorm only applies to AMP_FP16 and AMP_BF16 precisions.'))
        return model

    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.GroupNorm: to_LPGroupNorm}

    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(NoEffectWarning('No instances of torch.nn.GroupNorm found.'))
    log.info(f'Successfully replaced {len(replaced_instances)} instances of GroupNorm with LowPrecisionGroupNorm')


class LowPrecisionGroupNorm(Algorithm):
    """
    Replaces all instances of `torch.nn.GroupNorm` with `composer.algorithms.low_precision_groupnorm.low_precision_GroupNorm.LPGroupNorm`.

    LPGroupNorm is a thin wrapper around `torch.nn.GroupNorm` which forces the layer to run in lower precision (torch.float16 or torch.bfloat16)
    if autocast is enabled. This algorithm has no effect in FP32 or DeepSpeed FP16 mode, where autocast is disabled.

    This algorithm is intended to be used instead of Fused GroupNorm. They have similar behavior and performance.

    Args:
        apply_at (Event, optional): Event where algorithm is applied.
    """

    def __init__(self, apply_at: Optional[Event] = None):
        self.apply_at = Event.INIT if apply_at is None else apply_at
        if self.apply_at not in {Event.INIT, Event.AFTER_LOAD}:
            raise ValueError('LowPrecisionGroupNorm only supports application on Event.INIT and Event.AFTER_LOAD.')

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == self.apply_at

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        # VAE/CLIP are already in fp16, only apply to unet
        apply_low_precision_groupnorm(model=state.model.unet, optimizers=state.optimizers, precision=state._precision)
