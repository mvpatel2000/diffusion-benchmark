# Copyright 2022 MosaicML
# SPDX-License-Identifier: Apache-2.0

import argparse
import warnings
import textwrap

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


# Locally import algorithms. Normally, these would be imported from composer.algorithms but they
# are copied here for the benchmark to be more self-contained.
from ema import EMA
from low_precision_groupnorm import LowPrecisionGroupNorm as FusedGroupNorm
from fused_layernorm import FusedLayerNorm

from predict_speed_monitor import PredictSpeedMonitor
from data import StreamingLAIONDataset, SyntheticImageCaptionDataset, SyntheticLatentsDataset

try:
    import xformers
    is_xformers_installed = True
except:
    print('Warning: xformers is not installed.')
    is_xformers_installed = False


parser = argparse.ArgumentParser()

# Benchmark arguments
parser.add_argument('--disable_vae_clip', action='store_true')  # If set, use precomputed latents
parser.add_argument('--disable_unet', action='store_true')      # If set, only compute the latents

# Dataloader arguments
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--num_samples', type=int, default=800000)
parser.add_argument('--remote', type=str)
parser.add_argument('--local', type=str, default='/tmp/mds-cache/mds-laion-2/')
parser.add_argument('--use_synth_data', action='store_true')

# Model argument
parser.add_argument('--model_name', type=str, default='stabilityai/stable-diffusion-2-base')
parser.add_argument('--use_fsdp_global_unet', action='store_true')
parser.add_argument('--use_fsdp_local_unet', action='store_true')

# Algorithm argument
parser.add_argument('--use_ema', action='store_true')
parser.add_argument('--use_fused_layernorm', action='store_true')
parser.add_argument('--use_fused_groupnorm', action='store_true')

# Logger arguments
parser.add_argument('--wandb_name', type=str)
parser.add_argument('--wandb_project', type=str)

# Trainer arguments
parser.add_argument('--device_train_microbatch_size', type=int)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
args = parser.parse_args()

class StableDiffusion(composer.models.ComposerModel):

    def __init__(self, 
        model_name: str = 'stabilityai/stable-diffusion-2-base', 
        use_vae_clip: bool = True, 
        use_unet: bool = True,
        use_fsdp_global_unet: bool = False,
        use_fsdp_local_unet: bool = False,
    ):
        super().__init__()
        self.use_vae_clip = use_vae_clip
        self.use_unet = use_unet
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')

        # Wrap the UNet in FSDP
        if use_fsdp_global_unet:
            print('global wrap')
            self.unet._fsdp_wrap = True
        if use_fsdp_local_unet:
            print('local wrap')
            for up_block in self.unet.up_blocks:
                up_block._fsdp_wrap = True
            self.unet.mid_block._fsdp_wrap = True
            for up_block in self.unet.up_blocks:
                up_block._fsdp_wrap = True

        # Optionally load VAE/CLIP for preprocessing
        if self.use_vae_clip:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae', torch_dtype=torch.float16)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16)

            # Freeze vae and text_encoder when training
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)

    def forward(self, batch):
        if self.use_vae_clip:
            images, captions = batch['image'], batch['caption']
            # Run the VAE and CLIP in fp16 as it is inference only
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                # Encode the images to the latent space.
                latents = self.vae.encode(images.half())['latent_dist'].sample().data
                # Magical scaling number (See https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515)
                latents *= 0.18215

                # Encode the text. Assumes that the text is already tokenized
                conditioning = self.text_encoder(captions)[0]  # Should be (batch_size, 77, 768)
        else:
            latents, conditioning = batch['latents'], batch['conditioning']

        if self.use_unet:
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

    # Validate params
    if args.disable_vae_clip and args.disable_unet:
        raise ValueError('Cannot disable both VAE/CLIP and UNet')
    if args.use_fsdp_global_unet and args.use_fsdp_local_unet:
        raise ValueError('Cannot use both global and local UNet FSDP')

    model = StableDiffusion(
        model_name=args.model_name,
        use_vae_clip=not args.disable_vae_clip,
        use_unet=not args.disable_unet,
        use_fsdp_global_unet=args.use_fsdp_global_unet,
        use_fsdp_local_unet=args.use_fsdp_local_unet,
    )

    # Set FSDP Config
    fsdp_config = None
    if args.use_fsdp_global_unet:
        fsdp_config = {
            'sharding_strategy': 'SHARD_GRAD_OP',
        }

    # Enable xformers memory efficient attention after model has moved to device. Otherwise,
    # xformers will leak memory on rank 0 and never clean it up for non-rank 0 processes.
    model = DeviceGPU().module_to_device(model)
    if is_xformers_installed:
        model.unet.enable_xformers_memory_efficient_attention()
        if not args.disable_vae_clip:
            model.vae.enable_xformers_memory_efficient_attention()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    lr_scheduler = composer.optim.ConstantScheduler()

    device_batch_size = args.batch_size // dist.get_world_size()
    # Override batch size with microbatch size since we don't have microbatching when using predict
    if args.disable_unet:
        if args.device_train_microbatch_size * dist.get_world_size() != device_batch_size:
            warnings.warn(textwrap.dedent(
                f'`device_train_microbatch_size` ({args.device_train_microbatch_size}) * num_gpus ({dist.get_world_size()}) '
                f'!= `batch_size` ({args.batch_size}), which must be equal when calling `predict` as predict does not '
                 'microbatch. Ignoring `batch_size` and using `device_train_microbatch_size` instead.'))
        device_batch_size = args.device_train_microbatch_size

    sampler = None
    if args.use_synth_data:
        if args.disable_vae_clip:
           train_dataset = SyntheticLatentsDataset(image_size=args.image_size, num_samples=args.num_samples)
        else:
            train_dataset = SyntheticImageCaptionDataset(image_size=args.image_size, num_samples=args.num_samples)
        sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)
    else:
        if args.disable_vae_clip:
            raise ValueError('`--disable_vae_clip` is only valid when using synthetic data `--use_synth_data`')
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

    algorithms = []
    if args.use_ema:
        algorithms.append(EMA())
    if args.use_fused_layernorm:
        algorithms.append(FusedLayerNorm())
    if args.use_fused_groupnorm:
        algorithms.append(FusedGroupNorm())

    speed_monitor = None
    if args.disable_unet:
        speed_monitor = PredictSpeedMonitor(window_size=100)  # Predict requires different events
    else:
        speed_monitor = SpeedMonitor(window_size=100)

    callbacks = [
        speed_monitor,
        MemoryMonitor(),
    ]

    logger = composer.loggers.WandBLogger(name=args.wandb_name, project=args.wandb_project)

    device_train_microbatch_size = 'auto'
    if args.device_train_microbatch_size:
        device_train_microbatch_size = args.device_train_microbatch_size

    trainer = composer.Trainer(
        model=model,
        fsdp_config=fsdp_config,
        train_dataloader=train_dataloader,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=algorithms,
        callbacks=callbacks,
        loggers=logger,
        max_duration='1ep',
        device_train_microbatch_size=device_train_microbatch_size,
    )

    # Train the model!
    if not args.disable_unet:
        trainer.fit()
    # If UNet is disabled, only run inference to measure throughput of computing latents
    else:
        trainer.predict(dataloader=train_dataloader, return_outputs=False)

if __name__ == "__main__":
    print(args)
    main(args)
