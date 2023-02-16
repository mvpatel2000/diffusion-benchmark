import time
import torch
from diffusers import AutoencoderKL
from composer.devices import DeviceGPU

model_name: str = 'stabilityai/stable-diffusion-2-base'
vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
vae = DeviceGPU().module_to_device(vae)
vae.requires_grad_(False)

images = torch.randn(8, 3, 512, 512, device='cuda:0')

vae_q = torch.quantization.quantize_dynamic(
    vae,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8
)  # the target dtype for quantized weights

out = vae_q.encode(images)['latent_dist'].sample().data

# Burn in
for _ in range(2):
    vae.encode(images)['latent_dist'].sample().data
torch.cuda.synchronize()

start_time = time.time()
N = 10
for i in range(N):
    iter_time = time.time()
    vae.encode(images)['latent_dist'].sample().data
    torch.cuda.synchronize()
    print(f'VAE iter {i} time: {time.time() - iter_time}')
print(f'VAE time: {(time.time() - start_time)/N}')

start_time = time.time()
N = 10
for i in range(N):
    iter_time = time.time()
    vae_q.encode(images)['latent_dist'].sample().data
    torch.cuda.synchronize()
    print(f'VAE Quantized iter {i} time: {time.time() - iter_time}')
print(f'VAE Quantized time: {(time.time() - start_time)/N}')