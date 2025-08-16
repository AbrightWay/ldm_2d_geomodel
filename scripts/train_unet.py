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
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils_main import *
# Monai and diffusers modules
import monai
from monai.utils import first, set_determinism
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from rich.console import Console
from rich.table import Table
set_determinism(42)

# Set directories
parser = argparse.ArgumentParser(description="VAE Training")
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
parser.add_argument(
    "-n",
    "--experiment_name",
    default="SD_f8_c64_r1",
    help="Experiment name to be used for saving the trained Unet model and logs",
)
args = parser.parse_args()
env_dict = json.load(open(args.environment_file, "r"))
config_dict = json.load(open(args.config_file, "r"))
for k, v in env_dict.items():
    setattr(args, k, v)
for k, v in config_dict.items():
    setattr(args, k, v)

# Print loaded configurations
console = Console()
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Configuration", style="dim", width=30)
table.add_column("Value")

args_dict = vars(args)
for key, value in args_dict.items():
    if isinstance(value, dict):
        # For nested dictionaries, create a sub-table or format nicely
        sub_table = Table(show_header=False, box=None)
        for sub_key, sub_value in value.items():
            sub_table.add_row(f"[cyan]{sub_key}[/cyan]", str(sub_value))
        table.add_row(f"[bold]{key}[/bold]", sub_table)
    else:
        table.add_row(key, str(value))

console.print(table)
# Set directories

if not os.path.exists(args.trained_unet_dir):
    os.makedirs(args.trained_unet_dir)
    
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
m_train_loader, m_val_loader, _ = prepare_conditional_geomodel_dataset(
    args,
    batch_size=args.diffusion_train['batch_size'],
    well_facies=args.diffusion_train['well_facies'],
    prob_maps=args.diffusion_train['prob_maps'],
    augmentation_level=args.diffusion_train['augmentation_level'],
    condition_level=args.diffusion_train['condition_level']
)

first_batch,first_prob_maps,first_well_facies = next(iter(m_train_loader))
first_val_batch,first_prob_maps_val,first_well_facies_val = next(iter(m_val_loader))
print(f"First batch shape: {first_batch.shape}, with {len(m_train_loader)} batches in train_loader")
print(f"First prob_maps shape: {first_prob_maps.shape}, with {len(m_train_loader)} batches in train_loader")
print(f"First validation batch shape: {first_val_batch.shape}, with {len(m_val_loader)} batches in val_loader")
print(f"First validation prob_maps shape: {first_prob_maps_val.shape}, with {len(m_val_loader)} batches in val_loader")
# Initiate variational autoencoder (VAE) model and load pre-trained weights
autoencoderkl = define_instance(args, "autoencoder_def").to(device)
# Use the VAE checkpoint specified in the autoencoder_train config
vae_checkpoint_path = args.diffusion_train.get('vae_ckpt')
if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
    print(f"Loading VAE from checkpoint: {vae_checkpoint_path}")
    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    autoencoderkl.load_state_dict(checkpoint["model"], strict=False)
else:
    raise ValueError("A pre-trained VAE checkpoint must be specified in the config file.")
autoencoderkl.eval()

# Initiate U-net and condition encoder models
unet = define_instance(args, "diffusion_def").to(device)

# Calculate number of input channels for the condition encoder
c_in = 0
if args.diffusion_train.get('prob_maps', False):
    c_in += 1  # for prob_maps
if args.diffusion_train.get('well_facies', False):
    c_in += 4  # for well_facies (n_facies=3 + 1 for empty)
if c_in == 0:
    # If no conditioning, effnet is not needed
    effnet = None
    print("Training UNet without conditioning.")
else:
    print(f"Condition encoder will have {c_in} input channels.")
    effnet = EfficientNetEncoder(
        c_latent=args.condition_encoder['c_latent'],
        effnet=args.condition_encoder['effnet'],
        context_dim=args.condition_encoder['context_dim'],
        c_in=c_in
    ).to(device)

# Set noise scheduler
scheduler = DDPMScheduler(num_train_timesteps=args.NoiseScheduler['num_train_timesteps'], schedule="linear_beta", beta_start=args.NoiseScheduler['beta_start'], beta_end=args.NoiseScheduler['beta_end'])

# Compute scaling factor
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(first_batch.to(device))
scale_factor = 1 / torch.std(z)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# Setup optimizer and scaler
trainable_params = list(unet.parameters())
if effnet:
    trainable_params.extend(list(effnet.parameters()))
optimizer = torch.optim.Adam(trainable_params, lr=args.diffusion_train['lr'])
scaler = torch.amp.GradScaler(device = device)

# Load checkpoint if specified
start_epoch = 0
if args.diffusion_train.get('ckpt') and os.path.exists(args.diffusion_train['ckpt']):
    print(f"Resuming from UNet checkpoint: {args.diffusion_train['ckpt']}")
    try:
        checkpoint = torch.load(args.diffusion_train['ckpt'], map_location=device)
        unet.load_state_dict(checkpoint["unet"])
        if effnet and "effnet" in checkpoint:
            effnet.load_state_dict(checkpoint["effnet"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
else:
    print("No UNet checkpoint found or specified. Starting training from scratch.")

# Training parameters
n_epochs = args.diffusion_train['max_epochs']
val_interval = args.diffusion_train['val_interval']
save_interval = args.diffusion_train['save_interval']

# Training loop
epoch_losses = []
val_losses = []
best_val_loss = float('inf')
device_str = "cuda" if device.type == "cuda" else "cpu"
start_time = time.time()

for epoch in range(start_epoch, start_epoch+n_epochs):
    unet.train()
    if effnet:
        effnet.train()
    
    epoch_loss = 0
    progress_bar = tqdm(enumerate(m_train_loader), total=len(m_train_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, (images, prob_rlzs, well_facies) in progress_bar:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_str, enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(images) * scale_factor
            
            conditioning = None
            if effnet:
                conditions_list = []
                if args.diffusion_train.get('prob_maps', False):
                    conditions_list.append(prob_rlzs.to(device))
                if args.diffusion_train.get('well_facies', False):
                    conditions_list.append(well_facies.to(device))
                
                if conditions_list:
                    combined_conditions = torch.cat(conditions_list, dim=1)
                    conditioning = effnet(combined_conditions)

            noise = torch.randn_like(z).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            
            noise_pred = inferer(
                inputs=images, 
                diffusion_model=unet, 
                noise=noise, 
                timesteps=timesteps,
                autoencoder_model=autoencoderkl,
                condition=conditioning
            )
            
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss (mse)": epoch_loss / (step + 1)})
        
    epoch_losses.append(epoch_loss / (step + 1))

    # --- Validation ---
    if (epoch + 1) % val_interval == 0:
        unet.eval()
        if effnet:
            effnet.eval()
        
        val_loss = 0
        with torch.no_grad():
            for val_step, (images, prob_rlzs, well_facies) in enumerate(m_val_loader, start=1):
                images = images.to(device)

                with torch.amp.autocast(device_str, enabled=True):
                    z = autoencoderkl.encode_stage_2_inputs(images) * scale_factor
                    
                    conditioning = None
                    if effnet:
                        conditions_list = []
                        if args.diffusion_train.get('prob_maps', False):
                            conditions_list.append(prob_rlzs.to(device))
                        if args.diffusion_train.get('well_facies', False):
                            conditions_list.append(well_facies.to(device))
                        
                        if conditions_list:
                            combined_conditions = torch.cat(conditions_list, dim=1)
                            conditioning = effnet(combined_conditions)

                    noise = torch.randn_like(z).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                    
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=unet,
                        autoencoder_model=autoencoderkl,
                        noise=noise,
                        timesteps=timesteps,
                        condition=conditioning,
                    )
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                val_loss += loss.item()
        
        val_loss /= (val_step+1)
        val_losses.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "unet": unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch
            }
            if effnet:
                checkpoint["effnet"] = effnet.state_dict()
            torch.save(checkpoint, f'{args.trained_unet_dir}/{args.experiment_name}_unet_best.pt')
            print(f"New best model saved with val loss: {best_val_loss:.4f} at epoch {epoch}")

    # --- Save Interval Checkpoint ---
    if (epoch + 1) % save_interval == 0:
        checkpoint = {
            "unet": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch
        }
        if effnet:
            checkpoint["effnet"] = effnet.state_dict()
        torch.save(checkpoint, f'{args.trained_unet_dir}/{args.experiment_name}_unet_epoch_{epoch + 1}.pt')

end_time = time.time()
print(f"Total training time: {(end_time - start_time)//3600}h {(end_time - start_time)%3600//60}m {(end_time - start_time)%60}s")


# --- Final Logging ---
train_logs = {
    "epoch_losses": epoch_losses,
    "val_losses": val_losses,
    "best_val_loss": best_val_loss,
    "total_time_seconds": end_time - start_time,
    "config": args_dict
}

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
with open(os.path.join(args.log_dir, f"{args.experiment_name}_unet_training_log.json"), "w") as f:
    json.dump(train_logs, f)
    
progress_bar.close()
print("Training finished and logs saved.")