import torch
import time
from apex.normalization.fused_layer_norm import FusedLayerNorm 


BURN_IN = 10
NUM_RUNS = 1000

# # Image Example
# N, C, H, W = 20, 300, 64, 64
# G = 1
# inp = torch.randn(N, C, H, W)

# # LayerNorm
# group_norm = torch.nn.GroupNorm(G, C)
# # print(group_norm.weight.shape, group_norm.bias.shape)
# out1 = group_norm(inp)
# out2 = group_norm(inp)
# assert torch.allclose(out1, out2)

# layer_norm = torch.nn.LayerNorm([C, H, W])
# # print(layer_norm.weight.shape, layer_norm.bias.shape)
# layer_norm.weight = torch.nn.Parameter(group_norm.weight.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W))
# layer_norm.bias = torch.nn.Parameter(group_norm.bias.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W))
# out3 = layer_norm(inp)
# assert torch.allclose(out1, out3)

# layer_norm = FusedLayerNorm([C, H, W])
# # print(layer_norm.weight.shape, layer_norm.bias.shape)
# layer_norm.weight = torch.nn.Parameter(group_norm.weight.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W))
# layer_norm.bias = torch.nn.Parameter(group_norm.bias.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W))
# out4 = layer_norm(inp)
# assert torch.allclose(out1, out4)

# # Benchmark
# for _ in range(BURN_IN):
#     group_norm(inp)
# torch.cuda.synchronize()
# start_time = time.time()
# for _ in range(NUM_RUNS):
#     group_norm(inp)
# torch.cuda.synchronize()
# print(f"GroupNorm: {time.time() - start_time}")

# for _ in range(BURN_IN):
#     layer_norm(inp)
# torch.cuda.synchronize() 
# start_time = time.time()
# for _ in range(BURN_IN):
#     layer_norm(inp)
# torch.cuda.synchronize()
# print(f"LayerNorm: {time.time() - start_time}")



# Image Example
N, C, H, W = 20, 300, 64, 64
G = 2
inp = torch.randn(N, C, H, W)

# LayerNorm
group_norm = torch.nn.GroupNorm(G, C)
# print(group_norm.weight.shape, group_norm.bias.shape)
out1 = group_norm(inp)
out2 = group_norm(inp)
assert torch.allclose(out1, out2)

layer_norm = torch.nn.LayerNorm([C//G, H, W], elementwise_affine=False)
# print(layer_norm.weight.shape, layer_norm.bias.shape)
# layer_norm.weight = torch.nn.Parameter(group_norm.weight.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W))
# layer_norm.bias = torch.nn.Parameter(group_norm.bias.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W))
out3 = layer_norm(inp.reshape(N*G, C//G, H, W)).reshape(N, C, H, W)
out3 = out3 * group_norm.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, -1, H, W) + group_norm.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, -1, H, W)
assert torch.allclose(out1, out3)


# Benchmark
for _ in range(BURN_IN):
    group_norm(inp)
torch.cuda.synchronize()
start_time = time.time()
for _ in range(NUM_RUNS):
    group_norm(inp)
torch.cuda.synchronize()
print(f"GroupNorm: {time.time() - start_time}")

for _ in range(BURN_IN):
    out = layer_norm(inp.reshape(N*G, C//G, H, W)).reshape(N, C, H, W)
    out = out * group_norm.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, -1, H, W) + group_norm.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, -1, H, W)
torch.cuda.synchronize() 
start_time = time.time()
for _ in range(BURN_IN):
    out = layer_norm(inp.reshape(N*G, C//G, H, W)).reshape(N, C, H, W)
    out = out * group_norm.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W) + group_norm.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W)
torch.cuda.synchronize()
print(f"LayerNorm: {time.time() - start_time}")