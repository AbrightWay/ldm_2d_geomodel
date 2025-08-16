import os
import json
import argparse
import time
import torch
import torch.nn.functional as F
import optuna
from tqdm import tqdm
import copy
import numpy as np
import random
from monai.utils import set_determinism
from generative.networks.nets import AutoencoderKL
from utils_main import prepare_conditional_geomodel_dataset
torch.backends.cudnn.benchmark = True

# --- Configuration and Setup ---
def setup_environment():
    """Parses arguments and loads configuration files."""
    parser = argparse.ArgumentParser(description="PyTorch VAE Optimization")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="../configs/environment.json",
        help="environment json file",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="../configs/config_train_16g.json",
        help="config json file with training hyperparameters",
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        type=int,
        default=100,
        help="number of trials for optuna optimization",
    )
    parser.add_argument(
        "-t",
        "--trial_epochs",
        type=int,
        default=150,
        help="number of epochs to train per trial",
    )
    
    args = parser.parse_args()
    
    with open(args.environment_file, "r") as f:
        env_dict = json.load(f)
    with open(args.config_file, "r") as f:
        config_dict = json.load(f)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    args.imgs_dir = os.path.join(project_root, 'data')
    
    return args, config_dict

# --- Optuna Objective Function ---
def objective(trial, args, device):
    """
    Defines a single trial for Optuna.
    Suggests hyperparameters, trains a model, and returns the validation loss.
    """
    # --- Load dataset ---
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    train_loader, val_loader, _ = prepare_conditional_geomodel_dataset(args,order = 'linear',batch_size=batch_size)
    first_batch = next(iter(train_loader))
    first_val_batch = next(iter(val_loader))
    print(f"First batch shape: {first_batch[0].shape}, Batch size: {batch_size} with {len(train_loader)} batches in train_loader")
    print(f"First validation batch shape: {first_val_batch[0].shape}, Batch size: {batch_size} with {len(val_loader)} batches in val_loader")
    # --- Hyperparameter Suggestions ---
    # Architecture
    val_interval = 2
    best_val_loss = 100.
    # n_levels = 4 # Fixed depth/ 8x downsampling
    start_channels = trial.suggest_categorical("start_channels", [16, 32, 64, 128])
    latent_channels = trial.suggest_int("latent_channels", 1,16)
    num_res_blocks = trial.suggest_int("num_res_blocks", 1, 4)
    channel_mults_ = trial.suggest_categorical(
        "channel_mults", ['1248', '1234', '1244', '1224', '1223']
    )
    channel_mults = [int(c) for c in channel_mults_]
    num_channels = [start_channels * mult for mult in channel_mults]
    

    # Training
    lr = 1e-4
    kl_weight = 1e-6
    norm_num_groups = 32 if min(num_channels) >=32 else min(num_channels)
    # --- Model and Optimizer ---
    try:
        model = AutoencoderKL(
            spatial_dims=2, in_channels=1, out_channels=1,
            num_channels=num_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(device = device)
    except Exception as e:
        del train_loader, val_loader  # Free memory
        torch.cuda.empty_cache()
        print(f"Error creating model for trial {trial.number}: {e}")
        return float('inf')
    device_str = "cuda" if device.type == "cuda" else "cpu"

    # --- Training Loop ---
    val_losses = []
    for epoch in range(args.trial_epochs):
        model.train()
        epoch_recon_loss = 0;epoch_kl_loss = 0;epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=125)
        progress_bar.set_description(f"Epoch {epoch}")
        for i, (batch, _, _) in progress_bar:
            images = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_str, enabled=True):
                reconstruction, z_mu, z_sigma = model(images)
                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                
                # Correct KL Loss calculation (matching train_vae.py)
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                
                loss = recons_loss + (kl_weight * kl_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_recon_loss += recons_loss.item()
            epoch_kl_loss += kl_loss.item() * kl_weight if kl_weight > 0 else 0.0
            epoch_loss += loss.item()
            progress_bar.set_postfix(
                {
                    "total_loss": epoch_loss / (i + 1),
                    "recons_loss": epoch_recon_loss / (i + 1),
                    "kl_loss": epoch_kl_loss / (i + 1),
                }
            )
        # --- Final Validation at the end of the trial ---
        if (epoch+1) % val_interval == 0:
            
            model.eval()
            val_loss = 0;val_recons_loss = 0;val_kl_loss = 0
            with torch.no_grad():
                for i, (batch, _, _) in enumerate(val_loader):
                    images = batch.to(device)
                    with torch.amp.autocast(device_str, enabled=True):
                        reconstruction, z_mu, z_sigma = model(images)
                        recons_loss = F.l1_loss(reconstruction.float(), images.float())
                        
                        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                        current_val_loss = recons_loss.item() + (kl_weight * kl_loss.item()) 
                    val_loss += current_val_loss
                    val_recons_loss += recons_loss.item()
                    val_kl_loss += kl_loss.item() * kl_weight if kl_weight > 0 else 0.0
                    
            final_val_loss = val_loss / (i + 1)
            val_recon_loss = val_recons_loss / (i + 1)
            val_kl_loss = val_kl_loss / (i + 1)
            
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
            print(f"Epoch {epoch}:")
            print(f" Total val loss: {final_val_loss}, Recon loss: {val_recon_loss}, KL loss: {val_kl_loss}, with best val loss: {best_val_loss}")
            # Pruning is implicitly handled by Optuna if the trial is stopped early.
            # We report the final loss once every val_interval epochs.
            val_losses.append(final_val_loss)
            trial.report(final_val_loss, epoch)
            if trial.should_prune():
                del model, optimizer, scaler, train_loader,val_loader  # Free memory
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
    del model, optimizer, scaler, train_loader,val_loader  # Free memory
    torch.cuda.empty_cache()
    return np.mean(val_losses[-10:])

# --- Save Optimal Config ---
def save_best_config(best_params, template_config, output_path):
    """Saves the best hyperparameters in the format of the original config file."""
    config = copy.deepcopy(template_config)
    
    # Update autoencoder architecture from best trial
    autoencoder_def = config.setdefault("autoencoder_def", {})

    # Retrieve hyperparameters from the best trial
    start_channels = best_params["start_channels"]
    latent_channels = best_params["latent_channels"]
    num_res_blocks_val = best_params["num_res_blocks"]
    channel_mults_ = best_params["channel_mults"]
    channel_mults = [int(c) for c in channel_mults_]
    batch_size = best_params["batch_size"]
    # Reconstruct derived values
    num_channels = [start_channels * mult for mult in channel_mults]
    num_res_blocks = num_res_blocks_val
    norm_num_groups = 32 if min(num_channels) >= 32 else min(num_channels)

    # Save to config dictionary
    autoencoder_def["start_channels"] = start_channels
    autoencoder_def["latent_channels"] = latent_channels
    autoencoder_def["num_channels"] = num_channels
    autoencoder_def["num_res_blocks"] = num_res_blocks
    autoencoder_def["norm_num_groups"] = norm_num_groups
    
    # Update training parameters from best trial
    autoencoder_train = config.setdefault("autoencoder_train", {})
    autoencoder_train['batch_size'] = batch_size
    # Note: 'lr' is fixed in the objective, so we don't save it from best_params.
    # autoencoder_train["kl_weight"] = best_params["kl_weight"]
    
    with open(output_path, "w") as f:
        json.dump(config, f)
    print(f"Best configuration saved to {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    seed = 1561
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_determinism(seed)
    args, template_config = setup_environment()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # train_loader, val_loader, _ = prepare_conditional_geomodel_dataset(args,order = 'linear')
    # print("Dataset loaded successfully.")

    study_name = f"vae_tuning_{args.n_trials}_trials_8x_prototype"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        sampler = optuna.samplers.TPESampler(seed=1561),
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,
        load_if_exists=True,
        storage=storage_name
        )
    
    start_time = time.time()
    try:
        study.optimize(
            lambda trial: objective(trial, args, device),
            n_trials=args.n_trials,
            #timeout=7200
        )
    except KeyboardInterrupt:
        print("Optimization stopped manually.")

    end_time = time.time()
    print(f"Total training time: {(end_time - start_time)//3600}h {(end_time - start_time)%3600//60}m {(end_time - start_time)%60}s")


    if study.best_trial:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value:.6f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Save results to a new structured config file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        tuned_config_path = os.path.join(project_root, "configs", "tune_8x_prototype.json")
        save_best_config(best_trial.params, template_config, tuned_config_path)
    else:
        print("No trials were completed.")