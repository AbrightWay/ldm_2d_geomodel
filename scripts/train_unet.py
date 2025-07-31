'''
File: train_unet.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a U-net to learn the de-noising process in the latent space of latent diffusion models
Note: requires Python package "monai" or "monai-generative" to load 2D U-net model and dataloaders
'''


# Import packages

# General imports
import os, json, time, argparse
import numpy as np
import shutil
import tempfile
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from PIL import Image 
import cv2
import matplotlib.pyplot as plt 

# Monai and diffusers modules
import monai
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from utils import *

set_determinism(42)

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "-e",
    "--environment-file",
    default="./config/environment.json",
    help="environment json file that stores environment path",
)
parser.add_argument(
    "-c",
    "--config-file",
    default="./config/config_train_16g.json",
    help="config json file that stores hyper-parameters",
)
args = parser.parse_args()
env_dict = json.load(open(args.environment_file, "r"))
config_dict = json.load(open(args.config_file, "r"))
for k, v in env_dict.items():
    setattr(args, k, v)
for k, v in config_dict.items():
    setattr(args, k, v)


# Set directories

if not os.path.exists(args.trained_unet_dir):
    os.makedirs(args.trained_unet_dir)
    
# Choose device
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load dataset
args.autoencoder_train['batch_size'] = args.diffusion_train['batch_size'] # Re-define the batch size for prepare_geomodels_dataset function as the function uses args.autoencoder_train['batch_size'] to define the batch size of the dataloaders
m_train_loader, m_val_loader = prepare_geomodels_dataset(args)




# Initiate variational autoendocder (VAE) model and load pre-trained weights
trained_vae_dir = args.trained_vae_dir
trained_vae_weights = trained_vae_dir + '/vae_epoch_1000_hd10.pt'

autoencoderkl = define_instance(args, "autoencoder_def").to(device)
checkpoint    = torch.load(trained_vae_weights)
autoencoderkl.load_state_dict(checkpoint)
autoencoderkl.eval()

# Initiate U-net model
unet = define_instance(args, "diffusion_def").to(device)


# Set noise scheduler to use for forward (noising) process
scheduler = DDPMScheduler(num_train_timesteps=args.NoiseScheduler['num_train_timesteps'], schedule="linear_beta", beta_start=args.NoiseScheduler['beta_start'], beta_end=args.NoiseScheduler['beta_end'])
#scheduler = DDIMScheduler(num_train_timesteps=100, schedule="linear_beta", beta_start=0.0001, beta_end=0.02)

# Compute scaling factor for non-perfectly Gaussian VAE latent spaces
example_data = first(m_train_loader)
device_str = "cuda" if device.type == "cuda" else "cpu"

with torch.no_grad():
    with torch.amp.autocast(device_str,enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(example_data["image"].to(device))

scale_factor = 1 / torch.std(z)


inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
optimizer = torch.optim.Adam(unet.parameters(), lr=args.diffusion_train['lr'])



# Training parameters
n_epochs      = args.diffusion_train['max_epochs']
val_interval  = args.diffusion_train['val_interval']
save_interval = args.diffusion_train['save_interval']

# Train the U-net on the noise predicting function

epoch_losses  = []
val_losses    = []
scaler        = torch.amp.GradScaler(device = device)
best_val_loss = 100.
start_time    = time.time()


for epoch in range(n_epochs):
    unet.train()
    autoencoderkl.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(m_train_loader), total=len(m_train_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_str,enabled=True):
            z_mu, z_sigma = autoencoderkl.encode(images)
            z = autoencoderkl.sampling(z_mu, z_sigma) 
            
            noise = torch.randn_like(z).to(device)
            
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = inferer(
                inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl
            )
            
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))
    
    # if (epoch + 1) % args.diffusion_train['save_interval'] == 0:
    #     torch.save(unet.state_dict(), f'{args.trained_unet_dir}' + f'/unet_epoch_{epoch + 1}.pt')

    if (epoch + 1) % val_interval == 0:
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(m_val_loader, start=1):
                images = batch["image"].to(device)

                with torch.amp.autocast(device_str,enabled=True):
                    z_mu, z_sigma = autoencoderkl.encode(images)
                    z = autoencoderkl.sampling(z_mu, z_sigma)

                    noise = torch.randn_like(z).to(device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                    ).long()
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=unet,
                        noise=noise,
                        timesteps=timesteps,
                        autoencoder_model=autoencoderkl,
                    )

                    loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()
        val_loss /= val_step
        val_losses.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(unet.state_dict(), f'{args.trained_unet_dir}' + f'/unet_best.pt')
        #     print(f"New best model saved with val loss: {best_val_loss:.4f} at epoch {epoch + 1}")
end_time = time.time()
print(f"Total training time: {(end_time - start_time)//3600}h {(end_time - start_time)%3600//60}m {(end_time - start_time)%60}s")
train_logs = {
    "epoch_losses": epoch_losses,
    "val_losses": val_losses,
    "best_val_loss": best_val_loss,
    "total_time": end_time - start_time
}
# with open(os.path.join(args.log_dir, f"unet_training_log_epochs{n_epochs}.json"), "w") as f:
#     json.dump(train_logs, f)
progress_bar.close()
