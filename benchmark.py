# Copyright 2022 MosaicML
# SPDX-License-Identifier: Apache-2.0

import argparse

import composer
import torch
import torch.nn.functional as F
from composer.utils import dist, reproducibility
from composer.devices import DeviceGPU
from composer.callbacks import MemoryMonitor, SpeedMonitor
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTextModel


from ema import EMA
from data import StreamingLAIONDataset, SyntheticImageCaptionDataset, SyntheticLatentsDataset

try:
    import xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False


parser = argparse.ArgumentParser()

# Dataloader arguments
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--num_samples', type=int, default=800000)
parser.add_argument('--remote', type=str)
parser.add_argument('--local', type=str, default='/tmp/mds-cache/mds-laion-2/')
parser.add_argument('--use_synth_data', action='store_true')
parser.add_argument('--use_latents', action='store_true')

# Model argument
parser.add_argument('--model_name', type=str, default='stabilityai/stable-diffusion-2-base')

# EMA argument
parser.add_argument('--use_ema', action='store_true')

# Logger arguments
parser.add_argument('--wandb_name', type=str)
parser.add_argument('--wandb_project', type=str)

# Trainer arguments
parser.add_argument('--device_train_microbatch_size', type=int)
args = parser.parse_args()

class StableDiffusion(composer.models.ComposerModel):

    def __init__(self, model_name: str = 'stabilityai/stable-diffusion-2-base', use_latents: bool = False):
        super().__init__()
        self.use_latents = use_latents
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
        if not self.use_latents:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')

        # Freeze vae and text_encoder when training
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def forward(self, batch):
        images, captions = batch['image'], batch['caption']

        if not self.use_latents:
            # Encode the images to the latent space.
            latents = self.vae.encode(images)['latent_dist'].sample().data
            # Magical scaling number (See https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515)
            latents *= 0.18215

            # Encode the text. Assumes that the text is already tokenized
            conditioning = self.text_encoder(captions)[0]  # Should be (batch_size, 77, 768)
        else:
            latents, conditioning = images, captions

        # Sample the diffusion timesteps
        timesteps = torch.randint(1, len(self.noise_scheduler), (latents.shape[0], ), device=latents.device)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Forward through the model
        return self.unet(noised_latents, timesteps, conditioning)['sample'], noise

    def loss(self, outputs, batch):
        return F.mse_loss(outputs[0], outputs[1])

    def get_metrics(self, is_train: bool):
        return None


def main(args):
    reproducibility.seed_all(17)

    model = StableDiffusion(model_name=args.model_name, use_latents=args.use_latents)

    # Enable xformers memory efficient attention after model has moved to device. Otherwise,
    # xformers will leak memory on rank 0 and never clean it up for non-rank 0 processes.
    model = DeviceGPU().module_to_device(model)
    if is_xformers_installed:
        model.unet.enable_xformers_memory_efficient_attention()
        model.vae.enable_xformers_memory_efficient_attention()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1.0e-4, weight_decay=0.001)
    lr_scheduler = composer.optim.ConstantScheduler()

    device_batch_size = args.batch_size // dist.get_world_size()

    sampler = None
    if args.use_synth_data:
        if args.use_latents:
           train_dataset = SyntheticLatentsDataset(image_size=args.image_size, num_samples=args.num_samples)
        else:
            train_dataset = SyntheticImageCaptionDataset(image_size=args.image_size, num_samples=args.num_samples)
        sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)
    else:
        if args.use_latents:
            raise ValueError('--use_latents is valid unless using synthetic data --use_synth_data')
        resize_transform = transforms.Resize((args.image_size, args.image_size))
        transform = transforms.Compose([resize_transform, transforms.ToTensor()])
        train_dataset = StreamingLAIONDataset(remote=args.remote,
                                              local=args.local,
                                              split=None,
                                              shuffle=True,
                                              transform=transform,
                                              batch_size=device_batch_size)


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

    ema = None
    if args.use_ema:
        ema = EMA()

    callbacks = [
        SpeedMonitor(window_size=100),
        MemoryMonitor(),
    ]

    logger = composer.loggers.WandBLogger(name=args.wandb_name, project=args.wandb_project)

    device_train_microbatch_size = 'auto'
    if args.device_train_microbatch_size:
        device_train_microbatch_size = args.device_train_microbatch_size

    trainer = composer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=ema,
        callbacks=callbacks,
        loggers=logger,
        max_duration='1ep',
        device_train_microbatch_size=device_train_microbatch_size,
    )
    trainer.fit()

if __name__ == "__main__":
    print(args)
    main(args)
