# Copyright 2022 MosaicML
# SPDX-License-Identifier: Apache-2.0

import argparse

import time
import composer
import torch
import torch.nn.functional as F
from composer.utils import dist, reproducibility
from composer.devices import DeviceGPU
from composer.callbacks import SpeedMonitor, MemoryMonitor
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from transformers import CLIPTextModel


from ema import EMA
from linearize_conv import LinearizeConv
from channels_last import ChannelsLast
from predict_speed_monitor import PredictSpeedMonitor

# Local Imports
# from fused_groupnorm import FusedGroupNorm
from low_precision_groupnorm import LowPrecisionGroupNorm as FusedGroupNorm
from fused_layernorm import FusedLayerNorm

# Composer Algorithms, applies to entire module...
# from composer.algorithms import LowPrecisionGroupNorm as FusedGroupNorm
# from composer.algorithms import LowPrecisionLayerNorm as FusedLayerNorm

# # Functional to UNet only
# from composer.core import Precision
# import composer.functional as cf

from data import SyntheticImageCaptionDataset, SyntheticLatentsDataset

try:
    import xformers
    is_xformers_installed = True
except:
    print('Warning: xformers is not installed.')
    is_xformers_installed = False

try:
    import bitsandbytes as bnb
    is_bitsandbytes_installed = True
except:
    print('Warning: bitsandbytes is not installed.')
    is_bitsandbytes_installed = False

parser = argparse.ArgumentParser()

# Benchmarking arguments
parser.add_argument('--synchronize', action='store_true')
parser.add_argument('--use_latents', action='store_true')
parser.add_argument('--compute_latents', action='store_true')

# Dataloader arguments
parser.add_argument('--batch_size', type=int, default=31)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--num_samples', type=int, default=100000)

# Model argument
parser.add_argument('--model_name', type=str, default='stabilityai/stable-diffusion-2-base')

# Algorithms argument
parser.add_argument('--use_ema', default=True)
parser.add_argument('--use_linear_conv', action='store_true')
parser.add_argument('--use_channels_last', action='store_true')
parser.add_argument('--use_fused_groupnorm', action='store_true')
parser.add_argument('--use_fused_layernorm', action='store_true')
parser.add_argument('--use_8bit_optimizer', action='store_true')

# Logger arguments
parser.add_argument('--wandb_name', type=str)
parser.add_argument('--wandb_project', type=str, default='stable-diffusion-microbenchmarking')

# Trainer arguments
parser.add_argument('--device_train_microbatch_size', type=int, default=8)
args = parser.parse_args()

class StableDiffusion(composer.models.ComposerModel):

    def __init__(self, model_name: str = 'stabilityai/stable-diffusion-2-base', use_latents: bool = False, synchronize: bool = False, compute_latents: bool = False):
        super().__init__()
        self.use_latents = use_latents
        self.synchronize = synchronize
        self.compute_latents = compute_latents
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
        if not self.use_latents:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae', torch_dtype=torch.float16)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16)

            # Freeze vae and text_encoder when training
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)

    def forward(self, batch):
        if self.synchronize:
            torch.cuda.synchronize()
        start_time = time.time()
        images, captions = batch['image'], batch['caption']

        if not self.use_latents:
            # Run the VAE and CLIP in fp16 as it is inference only
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                # Encode the images to the latent space.
                latents = self.vae.encode(images.half())['latent_dist'].sample().data
                # Magical scaling number (See https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515)
                latents *= 0.18215

                if self.synchronize:
                    torch.cuda.synchronize()
                print(f'VAE time: {time.time() - start_time}')
                start_time = time.time()

                # Encode the text. Assumes that the text is already tokenized
                conditioning = self.text_encoder(captions)[0]  # Should be (batch_size, 77, 768)
                if self.synchronize:
                    torch.cuda.synchronize()
                print(f'CLIP time: {time.time() - start_time}')
                start_time = time.time()
        else:
            latents, conditioning = images, captions

        if not self.compute_latents:
            # Sample the diffusion timesteps
            timesteps = torch.randint(1, len(self.noise_scheduler), (latents.shape[0], ), device=latents.device)
            # Add noise to the inputs (forward diffusion)
            noise = torch.randn_like(latents)
            noised_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            # Forward through the model
            out = self.unet(noised_latents, timesteps, conditioning)['sample'], noise
            if self.synchronize:
                torch.cuda.synchronize()
            print(f'Unet time: {time.time() - start_time}')
            return out

    def loss(self, outputs, batch):
        return F.mse_loss(outputs[0], outputs[1])

    def get_metrics(self, is_train: bool):
        return {}


def main(args):
    reproducibility.seed_all(17)

    model = StableDiffusion(
        model_name=args.model_name, 
        use_latents=args.use_latents, 
        synchronize=args.synchronize,
        compute_latents=args.compute_latents,
    )

    # Enable xformers memory efficient attention after model has moved to device. Otherwise,
    # xformers will leak memory on rank 0 and never clean it up for non-rank 0 processes.
    model = DeviceGPU().module_to_device(model)
    if is_xformers_installed:
        model.unet.enable_xformers_memory_efficient_attention()
        if not args.use_latents:
            model.vae.enable_xformers_memory_efficient_attention()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1.0e-4, weight_decay=0.001)
    if args.use_8bit_optimizer:
        optimizer = bnb.optim.AdamW8bit(params=model.parameters(), lr=1.0e-4, weight_decay=0.001)
    lr_scheduler = composer.optim.ConstantScheduler()

    device_batch_size = args.batch_size // dist.get_world_size()

    if args.use_latents:
        train_dataset = SyntheticLatentsDataset(image_size=args.image_size, num_samples=args.num_samples)
    else:
        train_dataset = SyntheticImageCaptionDataset(image_size=args.image_size, num_samples=args.num_samples)
    sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)


    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=device_batch_size,
        sampler=sampler,
        drop_last=True,
        prefetch_factor=2,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    algos = []
    if args.use_ema:
        algos.append(EMA())
    if args.use_linear_conv:
        algos.append(LinearizeConv())
    if args.use_fused_groupnorm:
        algos.append(FusedGroupNorm())
        # cf.apply_low_precision_groupnorm(model.unet, optimizer, Precision.AMP_FP16)
    if args.use_fused_layernorm:
        algos.append(FusedLayerNorm())
        # cf.apply_low_precision_layernorm(model.unet, optimizer, Precision.AMP_FP16)
    if args.use_channels_last:
        algos.append(ChannelsLast())

    callbacks = [
        SpeedMonitor(window_size=1) if not args.compute_latents else PredictSpeedMonitor(window_size=1),
        MemoryMonitor()
    ]
    print(callbacks)

    loggers = []
    if args.wandb_name is not None and args.wandb_project is not None:
        loggers = composer.loggers.WandBLogger(name=args.wandb_name, project=args.wandb_project)

    device_train_microbatch_size = 'auto'
    if args.device_train_microbatch_size:
        device_train_microbatch_size = args.device_train_microbatch_size

    trainer = composer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=algos,
        callbacks=callbacks,
        loggers=loggers,
        max_duration='1ep',
        device_train_microbatch_size=device_train_microbatch_size,
        train_subset_num_batches=10,
        progress_bar=False,
        log_to_console=True,
        console_log_interval='1ba'
    )

    if not args.compute_latents:
        trainer.fit()
    else:
        print('predicting')
        trainer.predict(dataloader=train_dataloader, subset_num_batches=20, return_outputs=False)

if __name__ == "__main__":
    print(args)
    main(args)
