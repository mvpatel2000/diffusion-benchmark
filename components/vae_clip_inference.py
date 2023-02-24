import torch
import time
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from transformers import CLIPTextModel

model_name: str = 'stabilityai/stable-diffusion-2-base'

vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae', torch_dtype=torch.float16).to('cuda:0')
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16).to('cuda:0')

images = torch.randn(8, 3, 512, 512, device='cuda:0', dtype=torch.float16)
captions = torch.randint(0, 128, (8, 77), dtype=torch.long, device='cuda:0')


# Burn in
for _ in range(2):
    vae.encode(images)['latent_dist'].sample().data
    text_encoder(captions)

# Benchmark
start_time = time.time()
for _ in range(20):
    iter_time = time.time()
    vae.encode(images)['latent_dist'].sample().data
    torch.cuda.synchronize()
    print(f'\tVAE time: {time.time() - iter_time}')
    iter_time = time.time()
    text_encoder(captions)
    torch.cuda.synchronize()
    print(f'\tCLIP time: {time.time() - iter_time}')
print(f'Elapsed time: {time.time() - start_time}')