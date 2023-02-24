import time
import torch

def benchmark(model, BURN_IN=10, NUM_ITER=1000, linear_model=False, name=''):
    if linear_model:
        data = torch.randn(8, 32*32, 640).cuda()
    else:
        data = torch.randn(8, 640, 32, 32).cuda()

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
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
    # torch.nn.Conv2d(640, 640, 1, 1),
).cuda()

linear_model = torch.nn.Sequential(
    torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
    # torch.nn.Linear(640, 640),
).cuda()

torch.backends.cuda.matmul.allow_tf32 = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(linear_model))
benchmark(linear_model, linear_model=True, name='Linear Model')

print(count_parameters(conv_model))
benchmark(conv_model, name='Conv Model')

