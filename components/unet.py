import time
import torch
from torch.profiler import ProfilerActivity, profile
from diffusers import UNet2DConditionModel, DDPMScheduler
from composer.devices import DeviceGPU

# from composer.core import Precision
# from composer import functional as cf
# from low_precision_groupnorm import apply_low_precision_groupnorm
# # from linearize_conv import apply_conv_linearization

model_name: str = 'stabilityai/stable-diffusion-2-base'
noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
unet = DeviceGPU().module_to_device(unet)

# # Speedup UNet
# cf.apply_fused_layernorm(unet, optimizers=None)
# apply_low_precision_groupnorm(unet, optimizers=None, precision=Precision.AMP_FP16)
# # apply_conv_linearization(unet, optimizers=None)
# unet.to(memory_format=torch.channels_last)

latents = torch.randn(8, 4, 32, 32, device='cuda:0')
conditioning = torch.randn(8, 77, 1024, device='cuda:0')

timesteps = torch.randint(1, len(noise_scheduler), (latents.shape[0], ), device=latents.device)
noise = torch.randn_like(latents)
noised_latents = noise_scheduler.add_noise(latents, noise, timesteps)

with torch.cuda.amp.autocast(True):
    # Burn in
    for _ in range(2):
        out = unet(noised_latents, timesteps, conditioning)['sample'], noise
    torch.cuda.synchronize()

    # start_time = time.time()
    # N = 10
    # for i in range(N):
    #     iter_time = time.time()
    #     out = unet(noised_latents, timesteps, conditioning)['sample'], noise
    #     torch.cuda.synchronize()
    #     print(f'UNet iter {i} time: {time.time() - iter_time}')
    # print(f'UNet time: {(time.time() - start_time)}')

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        out = unet(noised_latents, timesteps, conditioning)['sample'], noise
        out[0].mean().backward()
    torch.cuda.synchronize()
    print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total', row_limit=-1))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=-1))
