'''
File: train_vae.py
Author: Guido Di Federico (code is based on the implementation available at https://github.com/Project-MONAI/tutorials/tree/main/generative and https://github.com/huggingface/diffusers/)
Description: Script to train a variational autoencoder (VAE) to learn the mapping between geomodel space and low-dimensional latent space for latent diffusion models
Note: requires Python package "monai" or "monai-generative" to load VAE model and dataloaders
'''


# Import packages

# General imports
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
# Monai and diffusers modules

from monai.utils import set_determinism


# Set directories
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

if not os.path.exists(args.trained_vae_dir):
    os.makedirs(args.trained_vae_dir)
    
# Choose device
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(1561)


# Load dataset
m_train_loader, m_val_loader = prepare_geomodels_dataset(args)

# Set hard data conditioning points (first two coordinates are (x,y) points and third coordinate the pixel value)
hard_data_locations = np.array(args.autoencoder_train['hd_locs'])


# Initiate variational autoendocder (VAE) model
autoencoderkl = define_instance(args, "autoencoder_def").to(device)

# Train the VAE on three loss terms: (1) reconstruction loss, (2) K-L divergence loss, (3) hard data facies loss
# Training parameters
n_epochs      = args.autoencoder_train['max_epochs']
val_interval  = args.autoencoder_train['val_interval']
save_interval = args.autoencoder_train['save_interval']
kl_weight     = args.autoencoder_train['kl_weight']
hd_weight     = args.autoencoder_train['hd_weight']

# Gradient parameters (optimizer and scaler)
optimizer = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler(device = device)

# Training loop

epoch_losses, log_recons_losses, log_kl_losses, log_hd_losses = [], [], [], []
val_losses   = []
device_str = "cuda" if device.type == "cuda" else "cpu"
start_time = time.time()
for epoch in range(n_epochs):
        
    autoencoderkl.train()
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_hd_loss = 0
    epoch_loss = 0
    progress_bar = tqdm(enumerate(m_train_loader), total=len(m_train_loader), ncols=200)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        m_batch = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_str,enabled=True):
            
            reconstruction, z_mu, z_sigma = autoencoderkl(m_batch)
            recons_loss = F.l1_loss(reconstruction.float(), m_batch.float())

            hd_loss =  hard_data_loss_func(reconstruction, m_batch, hard_data_locations) if hd_weight > 0 else 0.0

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            
            loss_tot = recons_loss + (kl_weight * kl_loss) + (hd_weight * hd_loss)


        scaler.scale(loss_tot).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_recon_loss += recons_loss.item()
        epoch_kl_loss += kl_loss.item() * kl_weight if kl_weight > 0 else 0.0
        epoch_hd_loss += hd_loss.item() * hd_weight if hd_weight > 0 else 0.0
        epoch_loss += loss_tot.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_recon_loss / (step + 1),
                "kl_loss": epoch_kl_loss / (step + 1),
                "hd_loss": epoch_hd_loss / (step + 1),
            }
        )
    
    epoch_losses.append(epoch_loss / (step + 1))
    log_recons_losses.append(epoch_recon_loss / (step + 1))
    log_kl_losses.append(epoch_kl_loss / (step + 1))
    log_hd_losses.append(epoch_hd_loss / (step + 1))
    
    hd_str = str(args.autoencoder_train['hd_weight']).replace(".","") if hd_weight <1 else str(int(args.autoencoder_train['hd_weight']))
    if (epoch + 1) % save_interval == 0:
        torch.save(autoencoderkl.state_dict(), f'{args.trained_vae_dir}' + f'/vae_epoch_{epoch + 1}_hd{hd_str}.pt')

    if (epoch + 1) % val_interval == 0:
        autoencoderkl.eval()
        val_loss, val_recon_loss, val_kl_loss, val_hd_loss = 0, 0, 0, 0
        with torch.no_grad():
            for val_step, batch in enumerate(m_val_loader, start=1):
                m_batch = batch["image"].to(device)

                with torch.amp.autocast(device_str,enabled=True):
                    reconstruction, z_mu, z_sigma = autoencoderkl(m_batch)
                    recons_loss = F.l1_loss(reconstruction.float(), m_batch.float())
                    val_recon_loss += recons_loss.item()
                    
                    hd_loss =  hard_data_loss_func(reconstruction, m_batch, hard_data_locations) if hd_weight > 0 else 0.0
                    val_hd_loss += hd_loss.item() * hd_weight if hd_weight > 0 else 0.0
                    
                    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                    val_kl_loss += kl_loss.item() * kl_weight if kl_weight > 0 else 0.0
                    
                    loss_g = recons_loss + (kl_weight * kl_loss) + (hd_weight * hd_loss)


                val_loss += loss_g.item()
        val_loss /= val_step
        val_recon_loss /= val_step
        val_kl_loss /= val_step
        val_hd_loss /= val_step
        
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}:")
        print(f" Total val loss: {val_loss}, Recon loss: {val_recon_loss}, KL loss: {val_kl_loss}, HD loss: {val_hd_loss}")
        if val_loss < min(val_losses):
            torch.save(autoencoderkl.state_dict(), f'{args.trained_vae_dir}' + f'/vae_epoch_{epoch + 1}_hd{hd_str}_best.pt')
            print(f"Best model saved at epoch {epoch + 1} with val loss: {val_loss}")
train_logs = {
    "epoch_losses": epoch_losses,
    "log_recons_losses": log_recons_losses,
    "log_kl_losses": log_kl_losses,
    "log_hd_losses": log_hd_losses,
    "val_losses": val_losses,
    "val_recon_losses": val_recon_loss,
    "val_kl_losses": val_kl_loss,
    "val_hd_losses": val_hd_loss,
    "total_time": time.time() - start_time
}

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
with open(os.path.join(args.log_dir, f"vae_training_log_epochs{n_epochs}_hd{hd_str}.json"), "w") as f:
    json.dump(train_logs, f)

progress_bar.close()
