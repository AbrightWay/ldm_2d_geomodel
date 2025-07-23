import torch, os 
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.bundle import ConfigParser
#set_determinism(42)




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