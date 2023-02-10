import time
import torch
import torch.nn.functional as F
import functools
from composer.utils import module_surgery


class LinearConv(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight[:, :, 0, 0].contiguous()
        self.bias = bias

    def forward(self, x):
        # print(x.stride())
        # 0.91
        # return F.linear(x.permute(0, 2, 3, 1).contiguous(), self.weight, self.bias).permute(0, 3, 1, 2).contiguous()
        # 0.60
        return F.linear(x.permute(0, 2, 3, 1), self.weight, self.bias).permute(0, 3, 1, 2)

def apply_conv1x1(model: torch.nn.Module) -> torch.nn.Module:
    transforms = {
        torch.nn.Conv2d: functools.partial(
            _maybe_replace_conv2d,
        )
    }
    module_surgery.replace_module_classes(model, optimizers=None, policies=transforms)

    return model

def _maybe_replace_conv2d(
    module: torch.nn.Conv2d,
    module_index: int,
):
    if module.kernel_size == 1 or module.kernel_size == (1, 1):
        return LinearConv(module.weight, module.bias)
    return None



def benchmark(model, BURN_IN=10, NUM_ITER=100, linear_model=False, name=''):
    if linear_model:
        data = torch.randn(8*32*32, 640).cuda()
    else:
        data = torch.randn(8, 640, 32, 32).cuda()

    # Benchmark model. Run one forward pass to load in caches, and then take average of 10 runs.
    for i in range(BURN_IN):
        model(data)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(NUM_ITER):
        model(data)
    torch.cuda.synchronize()
    print(f'[{name}] Time:\t {(time.time() - start_time)}')

conv_model = torch.nn.Sequential(
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
    torch.nn.Conv2d(640, 640, 1, 1),
).cuda()

linear_model = torch.nn.Sequential(
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
    torch.nn.Linear(640, 640),
).cuda()

benchmark(linear_model, linear_model=True, name='Linear Model')

benchmark(conv_model, name='Conv Model')

benchmark(apply_conv1x1(conv_model), name='Conv Model 1x1')

