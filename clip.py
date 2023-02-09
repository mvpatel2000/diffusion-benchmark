import time
import torch
from transformers import CLIPTextModel
from composer.devices import DeviceGPU

model_name: str = 'stabilityai/stable-diffusion-2-base'
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
text_encoder = DeviceGPU().module_to_device(text_encoder)
text_encoder.requires_grad_(False)

caption_length = 77
batch_size = 8
captions = torch.randint(0, 128, (batch_size, caption_length), dtype=torch.long, device='cuda:0')

# Burn in
for _ in range(2):
    text_encoder(captions)

start_time = time.time()
N = 10
for i in range(N):
    iter_time = time.time()
    text_encoder(captions)[0]
    print(f'CLIP iter {i} time: {time.time() - iter_time}')
print(f'CLIP time: {(time.time() - start_time)/N}')