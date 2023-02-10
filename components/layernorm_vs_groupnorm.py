import torch
from apex.normalization.fused_layer_norm import FusedLayerNorm as APEXFusedLayerNorm
import time

BURN_IN = 10
NUM_SAMPLES = 300

# Image Example
N, C, H, W = 20, 300, 64, 64
G = 10
inp = torch.randn(N, C, H, W)

# LayerNorm
layer_norm = torch.nn.LayerNorm([C, H, W])

for _ in range(BURN_IN):
    output = layer_norm(inp)
torch.cuda.synchronize()
start_time = time.time()
for _ in range(NUM_SAMPLES):
    output = layer_norm(inp)
torch.cuda.synchronize()
print(f'LayerNorm time: {time.time() - start_time}')

# FusedLayerNorm
fused_layer_norm = APEXFusedLayerNorm(normalized_shape=layer_norm.normalized_shape, eps=layer_norm.eps)

for _ in range(BURN_IN):
    output = fused_layer_norm(inp)
torch.cuda.synchronize()
start_time = time.time()
for _ in range(NUM_SAMPLES):
    output = fused_layer_norm(inp)
torch.cuda.synchronize()
print(f'FusedLayerNorm time: {time.time() - start_time}')

# GroupNorm
group_norm = torch.nn.GroupNorm(G, C)

for _ in range(BURN_IN):
    output = group_norm(inp)
torch.cuda.synchronize()
start_time = time.time()
for _ in range(NUM_SAMPLES):
    output = group_norm(inp)
torch.cuda.synchronize()
print(f'GroupNorm time: {time.time() - start_time}')