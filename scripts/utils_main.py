import torch, os 
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.bundle import ConfigParser
import torch.nn as nn
import torchvision
from utils_GANSim import *
import numpy as np
set_determinism(42)




def hard_data_loss_func(reconstruction, m_batch, hard_data_locations):
    reconstruction_hd = [reconstruction[...,loc[0],loc[1]] for loc in hard_data_locations]
    reconstruction_hd_vector =  torch.stack(reconstruction_hd, dim=0).flatten()
    m_batch_hd = [m_batch[...,loc[0],loc[1]] for loc in hard_data_locations]
    m_batch_hd_vector = torch.stack(m_batch_hd, dim=0).flatten()
    hd_loss =  F.mse_loss(m_batch_hd_vector, reconstruction_hd_vector)
    return hd_loss


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def prepare_geomodels_dataset(args, train_split:float=0.7, val_split:float=0.2):
    geomodels_dataset = [{"image": os.path.join(args.imgs_dir, img)} for img in os.listdir(args.imgs_dir)][:4000]
    N_data = len(geomodels_dataset)
    

    # Split dataset
    test_split        = 1 - train_split - val_split
    batch_size        = args.autoencoder_train['batch_size']

    m_train_list    = geomodels_dataset[:int(N_data*train_split)]
    m_val_list      = geomodels_dataset[int(len(m_train_list)):int(N_data*(1-test_split))+1]
    m_test_list     = geomodels_dataset[int(-N_data*test_split):]

    # Transform dataset

    # Training set
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True)]
    )

    m_train_ds = Dataset(data=m_train_list, transform=train_transforms)
    m_train_loader = DataLoader(m_train_ds, batch_size=batch_size, shuffle=True)

    # Validation set
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )
    m_val_ds = Dataset(data=m_val_list, transform=val_transforms)
    m_val_loader = DataLoader(m_val_ds, batch_size=batch_size, shuffle=True)

    # Testing set
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    m_test_ds = Dataset(data=m_test_list, transform=val_transforms)
    m_test_loader = DataLoader(m_test_ds, batch_size=batch_size, shuffle=True)
    return m_train_loader, m_val_loader

def unique_tensors_indicies(input_tensor):
    """
    Returns the unique tensors and their first occurrence indices from a batch of tensors.
    
    """
    unique_tensors, inverse_indices = torch.unique(input_tensor.view(input_tensor.shape[0], -1), dim=0, return_inverse=True)
    perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
    inverse_indices_sorted, perm_sorted = inverse_indices.sort()
    unique_indices = perm_sorted[torch.cat([torch.tensor([True], device=input_tensor.device), inverse_indices_sorted[1:] != inverse_indices_sorted[:-1]])]
    return unique_tensors, unique_indices
    
    
def prepare_conditional_geomodel_dataset(args=None, batch_size:int=None, well_facies:bool = True, prob_maps:bool = True, augmentation_level:int = 0,condition_level:int = 1):
    """_summary_

    Args:
        args (_type_, optional): parsed argument in the main file. Defaults to None.
        batch_size (int, optional): . Defaults to None.
        well_facies (bool, optional): well facies condition [N,n_facies+1,H,W], i.e. [N,4,64,64]. Defaults to False.
        prob_maps (bool, optional): probability maps condition [N,1,H,W], i.e. [N,1,64,64]. Defaults to False.
        augmentation_level (int, optional): multiplier data size (0->3). Defaults to 0.
        condition_level (int, optional): multiplier conditions (1->3). Defaults to 1.

    Returns:
        dataloaders
    """
    # Load dataset
    train_dataset     = torch.load(os.path.join(args.imgs_dir, "guido_train_dataset_augmented_with_conditions.pt"))
    val_dataset = torch.load(os.path.join(args.imgs_dir, "guido_val_dataset_augmented_conditions.pt"))
    test_dataset   = torch.load(os.path.join(args.imgs_dir, "guido_test_dataset_augmented_conditions.pt"))
    
    # Define train components
    train_images = train_dataset[augmentation_level][condition_level-1][0]
    train_prob_maps = train_dataset[augmentation_level][condition_level-1][1][:, :1] if prob_maps else torch.zeros_like(train_images)
    train_well_facies = train_dataset[augmentation_level][condition_level-1][1][:, 1:] if well_facies else torch.zeros_like(train_images)
    # Define validation components
    val_images = val_dataset[condition_level-1][0]
    val_prob_maps = val_dataset[condition_level-1][1][:, :1] if prob_maps else torch.zeros_like(val_images)
    val_well_facies = val_dataset[condition_level-1][1][:, 1:] if well_facies else torch.zeros_like(val_images)
    # Define test components
    test_images = test_dataset[condition_level-1][0]
    test_prob_maps = test_dataset[condition_level-1][1][:, :1] if prob_maps else torch.zeros_like(test_images)
    test_well_facies = test_dataset[condition_level-1][1][:, 1:] if well_facies else torch.zeros_like(test_images)
    print(f"Train images shape: {train_images.shape}, Train prob maps shape: {train_prob_maps.shape}, Train well facies shape: {train_well_facies.shape}")
    print(f"Validation images shape: {val_images.shape}, Validation prob maps shape: {val_prob_maps.shape}, Validation well facies shape: {val_well_facies.shape}")
    print(f"Test images shape: {test_images.shape}, Test prob maps shape: {test_prob_maps.shape}, Test well facies shape: {test_well_facies.shape}")
    del train_dataset, val_dataset, test_dataset  # Free memory
    
    # # Visualize some samples
    # n_plt = 10
    # indices = torch.randperm(train_images.shape[0])[:n_plt]  # Randomly permute indices for shuffling
    # show_grid(train_images[indices], n_samples = n_plt, n_cols= n_plt, title="Train Images")
    # show_grid(train_prob_maps[indices], n_samples = n_plt, n_cols= n_plt, title="Train Probability Maps")
    # indices = torch.randperm(val_images.shape[0])[:n_plt]  # Randomly permute indices for shuffling
    # show_grid(val_images[indices], n_samples = n_plt, n_cols= n_plt, title="Validation Images")
    # show_grid(val_prob_maps[indices], n_samples = n_plt, n_cols= n_plt, title="Validation Probability Maps")
    # indices = torch.randperm(test_images.shape[0])[:n_plt]  # Randomly permute indices for shuffling
    # show_grid(test_images[indices], n_samples = n_plt, n_cols= n_plt, title="Test Images")
    # show_grid(test_prob_maps[indices], n_samples = n_plt, n_cols= n_plt, title="Test Probability Maps")
    
    # # Data statistics
    # print(f"Statistics of train images: min={train_images.min()}, max={train_images.max()}, mean={train_images.mean()}, std={train_images.std()}")
    # print(f"Statistics of train prob maps: min={train_prob_maps.min()}, max={train_prob_maps.max()}, mean={train_prob_maps.mean()}, std={train_prob_maps.std()}")
    # print(f"Statistics of train well facies: min={train_well_facies.min()}, max={train_well_facies.max()}, mean={train_well_facies.mean()}, std={train_well_facies.std()}")
    # print(f"Statistics of val images: min={val_images.min()}, max={val_images.max()}, mean={val_images.mean()}, std={val_images.std()}")
    # print(f"Statistics of val prob maps: min={val_prob_maps.min()}, max={val_prob_maps.max()}, mean={val_prob_maps.mean()}, std={val_prob_maps.std()}")
    # print(f"Statistics of val well facies: min={val_well_facies.min()}, max={val_well_facies.max()}, mean={val_well_facies.mean()}, std={val_well_facies.std()}")
    # print(f"Statistics of test images: min={test_images.min()}, max={test_images.max()}, mean={test_images.mean()}, std={test_images.std()}")
    # print(f"Statistics of test prob maps: min={test_prob_maps.min()}, max={test_prob_maps.max()}, mean={test_prob_maps.mean()}, std={test_prob_maps.std()}")
    # print(f"Statistics of test well facies: min={test_well_facies.min()}, max={test_well_facies.max()}, mean={test_well_facies.mean()}, std={test_well_facies.std()}")
    
    #Wrap up into MONAI Dataset

    m_train_ds  = torch.utils.data.TensorDataset(train_images.float(), train_prob_maps.float(), train_well_facies.float())
    m_val_ds    = torch.utils.data.TensorDataset(val_images.float()  , val_prob_maps.float()  , val_well_facies.float())
    m_test_ds   = torch.utils.data.TensorDataset(test_images.float() , test_prob_maps.float(), test_well_facies.float())
    
    # Wrap up into MONAI DataLoader
    
    batch_size = batch_size if batch_size is not None else args.autoencoder_train['batch_size']
    
    m_train_loader  = torch.utils.data.DataLoader(m_train_ds, batch_size=batch_size, shuffle=True)
    m_val_loader    = torch.utils.data.DataLoader(m_val_ds,   batch_size=batch_size, shuffle=True)
    m_test_loader   = torch.utils.data.DataLoader(m_test_ds,  batch_size=batch_size, shuffle=True)
    
    return m_train_loader, m_val_loader, m_test_loader


class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16, effnet="efficientnet_b0", context_dim:int = 128,c_in:int = 1):
        
        super().__init__()
        self.conv_in = nn.Conv2d(c_in, 3, kernel_size=1, bias=False)  # Input conv layer to match EfficientNet input
        if effnet == "efficientnet_v2_s":
            self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features #22M params
        elif effnet == "efficientnet_v2_m":
            self.backbone = torchvision.models.efficientnet_v2_m(weights='DEFAULT').features # 54M params
        elif effnet == "efficientnet_v2_l":
            self.backbone = torchvision.models.efficientnet_v2_l(weights='DEFAULT').features # 120M params
        elif effnet == "efficientnet_b0":
            self.backbone = torchvision.models.efficientnet_b0(weights='DEFAULT').features # 5.3M params
        elif effnet == "efficientnet_b1":
            self.backbone = torchvision.models.efficientnet_b1(weights='DEFAULT').features # 7.8M params
        elif effnet == "efficientnet_b2":
            self.backbone = torchvision.models.efficientnet_b2(weights='DEFAULT').features # 9.2M params
        elif effnet == "efficientnet_b3":
            self.backbone = torchvision.models.efficientnet_b3(weights='DEFAULT').features # 12M params
        else:
            raise ValueError(f"Unsupported EfficientNet version: {effnet}")
        if effnet == "efficientnet_b2":
            effnet_out_channels = 1408
        elif effnet == "efficientnet_b3":
            effnet_out_channels = 1536
        else:
            effnet_out_channels = 1280  # Default for other EfficientNet versions 
            
        self.mapper = nn.Sequential(
            nn.Conv2d(effnet_out_channels, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )
        self.linear_layer = nn.Linear(c_latent, context_dim)  # Project to context dimension
        self.layer_norm = nn.LayerNorm(context_dim)
    def forward(self, x):
        x = self.conv_in(x)  # Convert single channel to 3 channels
        x = self.mapper(self.backbone(x))
        x = x.permute(0,2,3,1).view(x.size(0), -1, x.size(1)) # [B, H*W, C], consider every pixel of the latent space as a token
        x = self.linear_layer(x)  # Project to context dimension
        x = self.layer_norm(x)
        return x
    
def clamp_array(arr, min_val=0, max_val=255):
    
    if arr.min() == 0:
        return arr/arr.max() * (max_val - min_val) + min_val
    elif arr.min() == min_val and arr.max() == max_val:
        return arr
    else:
        return (arr - arr.min()) / (arr.max() - arr.min()) * (max_val - min_val) + min_val
    

def x_to_well_facies(x, n_facies:int=4,n_times:int = 1):
    """
    Convert facies model into well facies.
    Args:
        x (torch.Tensor): Input tensor of shape (B, 1, H, W) 
        n_facies (int): Number of facies classes.
    Returns:
        torch.Tensor: Well facies tensor of shape (B, H, W) with values in [0, n_facies-1].
        
    """
    X = img2facies(x,n_facies=n_facies).squeeze()
    facies_ind = np.unique(X)
    X_new = np.zeros((X.shape[0], len(facies_ind), X.shape[-2], X.shape[-1]))  # (b c h w)
    for i in facies_ind:
        X_new[:,i,:,:] = (X == i)
    X = X_new  # (b c h w)
    for ind in range(10,n_times+10):
        well_facies = []
        for i,facies in enumerate(X):
            x = np.argmax(facies, axis=0)  # (h w)
            x_masked = np.ones_like(x) * n_facies  # (h w)
            nr_of_wells = np.random.RandomState(42+i+ind).randint(1, 10)
            starting_height = np.random.RandomState(42+i+ind).uniform(10, 50)
            starting_point = np.random.RandomState(42+i+ind).randint(0, x.shape[0])
            a_list = np.random.RandomState(42+i+ind).uniform(-1, 1, nr_of_wells)
            for i in range(x_masked.shape[0]):
                for j in range(nr_of_wells):
                    y = (starting_height + i + 0.5) * a_list[j] + starting_point
                    if np.floor(y) >= 0 and np.floor(y) < x.shape[0]:
                        x_masked[i, int(np.floor(y))] = x[i, int(np.floor(y))]
            well_mat = np.zeros((facies.shape[0] + 1, facies.shape[1], facies.shape[2]))
            well_mat[n_facies, :, :] = np.ones_like(well_mat)[0, :, :]  # (c h w); the last channel refers to the "empty" space.
            for facies_type in np.unique(x_masked):
                well_mat[facies_type, :, :] = np.where(x_masked == facies_type, 1, 0)
            mask_loc = (x != x_masked)
            well_mat[n_facies, mask_loc] = 1
            well_mat[:n_facies, mask_loc] = 0
            well_facies.append(well_mat)
        well_facies = np.stack(well_facies)  # (b c h w)
        all_well_facies = well_facies if ind==10 else np.concatenate((all_well_facies, well_facies), axis=0)
    return all_well_facies # (b c h w) with c = n_facies +1, each channel is a one-hot encoding of the facies type.