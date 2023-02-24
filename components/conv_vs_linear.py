import time
import torch
import torch.nn.functional as F
import functools
from composer.utils import module_surgery


class LinearConv(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.layer = torch.nn.Linear(weight.shape[0], weight.shape[1])
        self.layer.weight = torch.nn.parameter.Parameter(weight[:, :, 0, 0].contiguous())
        self.layer.bias = torch.nn.parameter.Parameter(bias)

    def forward(self, x):
        # return F.linear(x.permute(0, 2, 3, 1), self.weight, self.bias).permute(0, 3, 1, 2)
        # return self.layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # return self.layer(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        # return self.layer(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)

        N, C_in, H, W = x.shape
        return self.layer(x.permute(0, 2, 3, 1).reshape(-1, C_in)).reshape(N, H, W, -1).permute(0, 3, 1, 2)
        
        # # print(x.shape)
        # out = self.layer(x)
        # # print(out.shape)
        # return out
        # return self.layer(x)

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



def benchmark(model, BURN_IN=10, NUM_ITER=1000, linear_model=False, name=''):
    data = torch.randn(8, 640, 32, 32, device=torch.device('cuda'))
    if linear_model:
        data = data.permute(0, 2, 3, 1).reshape(-1, 640).contiguous()
    else:
        data = data.to(memory_format=torch.channels_last)

    # Benchmark model. Run one forward pass to load in caches, and then take average of 10 runs.
    with torch.cuda.amp.autocast(True):
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.backends.cuda.matmul.allow_tf32 = True

print(count_parameters(linear_model))
benchmark(linear_model, linear_model=True, name='Linear Model')

conv_model.to(memory_format=torch.channels_last)

print(count_parameters(conv_model))
benchmark(conv_model, name='Conv Model')

conv_model = apply_conv1x1(conv_model)
print(count_parameters(conv_model))
benchmark(conv_model, name='Conv Model 1x1')

