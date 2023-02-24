import time
import torch
from diffusers import AutoencoderKL
from composer.devices import DeviceGPU

model_name: str = 'stabilityai/stable-diffusion-2-base'
vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
vae = DeviceGPU().module_to_device(vae)
vae.requires_grad_(False)

images = torch.randn(1, 3, 512, 512, device='cuda:0')

# Burn in
for _ in range(3):
    vae.encode(images)['latent_dist'].sample().data
torch.cuda.synchronize()

# Baseline
start_time = time.time()
N = 10
for i in range(N):
    iter_time = time.time()
    # Iter 1
    vae.encode(images)['latent_dist'].sample().data
    # Iter 2
    vae.encode(images)['latent_dist'].sample().data
    torch.cuda.synchronize()
    print(f'VAE iter {i} time: {time.time() - iter_time}')
print(f'Baseline VAE time: {(time.time() - start_time)}')

# Streams
device = torch.device(0)
s1 = torch.cuda.Stream(device=device)
s2 = torch.cuda.Stream(device=device)
start_time = time.time()
N = 10
for i in range(N):
    iter_time = time.time()
    # Iter 1
    with torch.cuda.stream(s1):
        vae.encode(images)['latent_dist'].sample().data
    # Iter 2
    with torch.cuda.stream(s2):
        vae.encode(images)['latent_dist'].sample().data
    torch.cuda.synchronize()
    print(f'VAE iter {i} time: {time.time() - iter_time}')
print(f'Streams VAE time: {(time.time() - start_time)}')