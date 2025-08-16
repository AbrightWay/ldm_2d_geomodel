import numpy as np
import scipy
from sklearn import manifold
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
RS = np.random.RandomState(42)
import matplotlib.lines as mlines
import pandas as pd
#import geostatspy.geostats as geostats                 # GSLIB methods convert to Python
#import geostatspy.GSLIB as GSLIB   
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

seed = 42

#-----------------np version-----------------------------------------------------------
def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] >= 1
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:S[1], -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.RandomState(seed).randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.RandomState(seed).randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]

#----------------------------------------------------------------------------

def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

#----------------------------------------------------------------------------

def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.RandomState(seed).randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats

#----------------------------------------------------------------------------

def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)

#----------------------------------------------------------------------------

gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0

def pyr_down(minibatch): # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]

def pyr_up(minibatch): # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')

def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid

def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch

def convert_to_matrix(a):
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n,dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype=float)
    out[mask] = a
    np.transpose(out)[mask] = a        
    return out      
    
    
    
#-----------------pytorch version------------------------------------------------------

gaussian_filter = torch.tensor([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]], dtype = torch.float32) / 256.0

def pyr_down_torch(minibatch):
    # minibatch: (N,C,H,W)  → reflect‐pad, convolve, then subsample
    N,C,H,W = minibatch.shape
    x = F.pad(minibatch, (2,2,2,2), mode='reflect')      # pad H/W dims
    # build weight: one 5×5 Gaussian per channel
    w = gaussian_filter.view(1,1,5,5).repeat(C,1,1,1).to(minibatch.device)
    blurred = F.conv2d(x, w, groups=C)
    return blurred[:,:,::2,::2]

def pyr_up_torch(minibatch):
    # minibatch: (N,C,h,w) → zero‐upsample, pad, convolve
    N,C,h,w = minibatch.shape
    up = torch.zeros((N,C,h*2,w*2), device=minibatch.device, dtype=minibatch.dtype)
    up[:,:,::2,::2] = minibatch
    x = F.pad(up, (2,2,2,2), mode='reflect')
    w = (gaussian_filter.view(1,1,5,5)*4.0).repeat(C,1,1,1).to(minibatch.device)
    return F.conv2d(x, w, groups=C)

def generate_laplacian_pyramid_torch(minibatch, num_levels):
    pyramid = [minibatch]
    for i in range(1, num_levels):
        pyramid.append(pyr_down_torch(pyramid[-1]))
        pyramid[-2] -= pyr_up_torch(pyramid[-1])
    return pyramid

def reconstruct_laplacian_pyramid_torch(pyr):
    x = pyr[-1]
    for level in reversed(pyr[:-1]):
        x = pyr_up_torch(x) + level
    return x

def get_descriptors_for_minibatch_torch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape  # (B, C, H, W)
    B, C, H, W = S
    N = nhoods_per_image * B
    Hh = nhood_size // 2
    device = minibatch.device

    # replace np.ogrid[0:N, 0:C, -Hh:Hh+1, -Hh:Hh+1]
    nhood = torch.arange(0, N, device=device, dtype=torch.long).view(N, 1, 1, 1)
    chan  = torch.arange(0, C, device=device, dtype=torch.long).view(1, C, 1, 1)
    x     = torch.arange(-Hh, Hh + 1, device=device, dtype=torch.long).view(1, 1, nhood_size, 1)
    y     = torch.arange(-Hh, Hh + 1, device=device, dtype=torch.long).view(1, 1, 1, nhood_size)

    img = nhood // nhoods_per_image
    # cx  = torch.randint(Hh, W - Hh, (N, 1, 1, 1), device=device)
    # cy  = torch.randint(Hh, H - Hh, (N, 1, 1, 1), device=device)
    # Compatibility with numpy:
    cx = torch.from_numpy(np.random.RandomState(seed).randint(Hh, W - Hh, size=(N, 1, 1, 1))).to(device)
    cy = torch.from_numpy(np.random.RandomState(seed).randint(Hh, H - Hh, size=(N, 1, 1, 1))).to(device)
    X = x + cx
    Y = y + cy

    idx = ((img * C + chan) * H + Y) * W + X     # shape (N, C, nhood_size, nhood_size)
    flat = minibatch.contiguous().view(-1)
    patches = flat[idx]    
    return patches

def finalize_descriptors_torch(desc):
    # desc: list of Tensors or a single Tensor
    if isinstance(desc, list):
        desc = torch.cat(desc, dim=0)
    # if shape is (N,C,h,w), normalize then flatten
    if desc.dim()==4:
        m = desc.mean(dim=(0,2,3), keepdim=True)
        s = desc.std(dim=(0,2,3), keepdim=True)
        desc = (desc - m) / s
        desc = desc.view(desc.shape[0], -1)
    return desc                                    # (N,descriptor_dim)

def sliced_wasserstein_torch(A, B, dir_repeats:int=4, dirs_per_repeat:int=64):
    # A,B: (N,D)
    results = []
    g = torch.Generator().manual_seed(seed)
    for _ in range(dir_repeats):
        #dirs = torch.randn(A.size(1), dirs_per_repeat, generator=g, device=A.device)
        dirs = torch.from_numpy(np.random.RandomState(seed).randn(A.shape[1], dirs_per_repeat)).float().to(A.device)
        dirs = dirs / dirs.norm(dim=0, keepdim=True)
        pA = A @ dirs                               # (N,dirs)
        pB = B @ dirs
        pA, _ = pA.sort(dim=0)
        pB, _ = pB.sort(dim=0)
        results.append((pA - pB).abs().mean())
    return torch.stack(results).mean()             # scalar

def convert_to_matrix_torch(a):
    n = int((len(a)*2)**0.5)+1
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)  # or torch.arange(n)[:,None] > torch.arange(n)
    out = torch.zeros((n, n), dtype=torch.float32)
    if isinstance(a, list):
        a = torch.tensor(a)
    out[mask] = a
    torch.transpose(out,1,0)[mask]  = a
    return out

#-----------------MSSWD MDS------------------------------------------------------



def MSSWD_MDS(imgs, num_groups:int = 300, num_images_per_group:int = 40, nhood_size:int = 5, nhoods_per_image:int = 32, dir_repeats:int = 4, dirs_per_repeat:int = 64, distance_metric:str = 'sliced_wasserstein'):
    """
    reals: list of real images (N,C,H,W), N = num_groups * num_images_per_group
    fakes: list of generated (fake) images
    num_groups: number of groups (e.g. 300 as described in the paper)
    num_images_per_group: number of images per group (e.g. 40 as described in the paper)
    nhood_size: patch size (e.g. 5 for 5x5 patches as described in the paper)
    nhoods_per_image: number of patches per image (e.g. 32 as described in the paper)
    dir_repeats: number of repeats for sliced wasserstein distance
    dirs_per_repeat: number of directions per repeat for sliced wasserstein distance
    distance_metric: 'sliced_wasserstein' or 'euclidean' (default is 'sliced_wasserstein')
    """
    if distance_metric == 'sliced_wasserstein':
        res = imgs.shape[2]
        resolutions = []
        while res >=16:
            resolutions.append(res)
            res //= 2
            
        groups_lap = []
        for i in range(num_groups):
            minibatch = imgs[i * num_images_per_group : (i + 1) * num_images_per_group]
            descriptors = [[] for res in resolutions]
            for lod, level in enumerate(generate_laplacian_pyramid_torch(minibatch, len(resolutions))):
                desc = get_descriptors_for_minibatch_torch(level, nhood_size, nhoods_per_image)
                descriptors[lod].append(desc)
            groups_lap.append(descriptors)

        kk = torch.tril(torch.ones((num_groups, num_groups)), -1)  
        coor = torch.argwhere(kk > 0)
        list_1 = coor[:, 0]
        list_2 = coor[:, 1]  
        
        gr_swd = []
        for gr in range(list_1.shape[0]):
            desc_1 = [finalize_descriptors_torch(d) for d in groups_lap[list_1[gr]]]
            desc_2 = [finalize_descriptors_torch(d) for d in groups_lap[list_2[gr]]]
            slw = [sliced_wasserstein_torch(dreal, dfake, dir_repeats, dirs_per_repeat) for dreal, dfake in zip(desc_1, desc_2)]
            gr_swd.append((torch.mean(torch.stack(slw)).cpu().item() * 1e3))
            

        swd_matrix = convert_to_matrix_torch(gr_swd)
    elif distance_metric == 'euclidean':
        scaler = StandardScaler()
        if len(imgs.shape) >1:
            imgs = imgs.reshape(imgs.shape[0], -1)  # Flatten the images if they are not already
        imgs = scaler.fit_transform(imgs)  # Normalize the images
        swd_matrix = pairwise_distances(imgs, metric='euclidean')
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1, random_state=seed)
    coos = mds.fit(swd_matrix.cpu().numpy()).embedding_
    
    return coos


#-----------------Auxiliary functions------------------------------------------------------


def cdf(a):
    if isinstance(a, list):
        a = np.asarray(a)
    x, counts = np.unique(a, return_counts=True)
    cummulative =  np.cumsum(counts)
    cummulative = cummulative / cummulative[-1]  # normalize to [0,1]
    return x, cummulative

def plot_MSSWD_MDS(real_imgs, fake_imgs, num_groups = 300, num_images_per_group = 40):
    """
    real_imgs: Pytorch tensor of real images (N,C,H,W)
    fake_imgs: Pytorch tensor of generated (fake) images (N,C,H,W)
    * Switch to Pytorch to use GPU
    """
    coos_real = MSSWD_MDS(real_imgs, num_groups, num_images_per_group)
    coos_fake = MSSWD_MDS(fake_imgs, num_groups, num_images_per_group)

    x_real_co, y_real_co = coos_real[:, 0], coos_real[:, 1]
    x_fake_co, y_fake_co = coos_fake[:, 0], coos_fake[:, 1]  
    
    
    plot_lim_min = -200
    plot_lim_max = 200
    # Create a figure with 6 plot areas
    fig, axes = plt.subplots(ncols=2, nrows=1, sharey='row')
    fig.set_size_inches(10, 4, forward=True)
    axes[0].set_title('Scatterplot')
    axes[0].set_xlim([plot_lim_min, plot_lim_max])
    axes[0].set_ylim([plot_lim_min, plot_lim_max])        
    axes[0].plot(x_real_co, y_real_co, 'ro', label = 'Real')
    axes[0].plot(x_fake_co, y_fake_co, 'b+', label = 'Generated')
    axes[0].legend(loc='upper right')
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 40

    np.random.seed(seed=seed)
    k_real = gaussian_kde((coos_real.T[:, :]))
    xi_real, yi_real = np.mgrid[plot_lim_min:plot_lim_max:nbins*1j, plot_lim_min:plot_lim_max:nbins*1j]
    zi_real = k_real(np.vstack([xi_real.flatten(), yi_real.flatten()]))
    axes[1].set_xlim([plot_lim_min, plot_lim_max])
    axes[1].set_ylim([plot_lim_min, plot_lim_max])                
    real_contr = axes[1].contour(xi_real, yi_real, zi_real.reshape(xi_real.shape), 6, colors='r') 

    np.random.seed(seed=seed)
    k_fake_prog = gaussian_kde((coos_fake.T[:, :]))
    xi_fake_prog, yi_fake_prog = np.mgrid[plot_lim_min:plot_lim_max:nbins*1j, plot_lim_min:plot_lim_max:nbins*1j]
    zi_fake_prog = k_fake_prog(np.vstack([xi_fake_prog.flatten(), yi_fake_prog.flatten()]))
    fake_contr_prog = axes[1].contour(xi_fake_prog, yi_fake_prog, zi_fake_prog.reshape(xi_fake_prog.shape), 5, colors='k', linestyles ='dashdot') 
    
    ## Calculate density for real images
    grid_x, grid_y = np.mgrid[plot_lim_min:plot_lim_max:100j, plot_lim_min:plot_lim_max:100j]
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density_real = k_real(positions).reshape(grid_x.shape)
    density_fake = k_fake_prog(positions).reshape(grid_x.shape)
    overlap = np.sum(np.minimum(density_real, density_fake))
    overlap_percentage = overlap / np.sum(density_real) * 100
    axes[1].text(0.95, 0.95, f'Overlap: {overlap_percentage:.2f}%', transform=axes[1].transAxes, fontsize=10, ha='left',va ='bottom')
    axes[1].set_title('Densityplot')
    # make proxy artists
    train_proxy = mlines.Line2D([], [], color='r', linestyle='solid', label='Real')
    prog_proxy  = mlines.Line2D([], [], color='k', linestyle='dashdot', label='Generated')

    # use them in your legend
    axes[1].legend(handles=[train_proxy, prog_proxy], loc='upper right')
    plt.show()
    return overlap_percentage


def plot_facies_cdf(real_imgs,fake_imgs, n_facies:int=3):
    """
    Plot the CDF of facies proportions for real and generated images.
    
    Parameters:
    - real_imgs: numpy array of real images (N, H, W, C)
    - fake_imgs: numpy array of generated images (N, H, W, C)
    - n_facies: number of facies classes
    """
    real_facies = [[] for _ in range(n_facies)]
    fake_facies = [[] for _ in range(n_facies)]
    
    for (real,fake) in zip(real_imgs, fake_imgs):
        freq_real, _ = np.histogram(real, bins = n_facies)
        freq_fake, _ = np.histogram(fake, bins = n_facies)
        for i in range(n_facies):
            real_facies[i].append(freq_real[i]/ np.sum(freq_real))
            fake_facies[i].append(freq_fake[i]/ np.sum(freq_fake))
    
    fig, axes = plt.subplots(ncols = n_facies, nrows=1, sharey='row', figsize=(15, 5))
    for i in range(n_facies):
        x_real, y_real = cdf(real_facies[i])
        x_fake, y_fake = cdf(fake_facies[i])
        axes[i].set_title(f'Facies {i+1}')
        axes[i].set_xlabel('Facies Proportion')
        axes[i].set_ylabel('CDF')
        axes[i].plot(x_real, y_real, 'r-', label='Real')
        axes[i].plot(x_fake, y_fake, 'b-', label='Generated')
        axes[i].legend(loc='upper left')
    plt.tight_layout()
    plt.show()

        
def img2facies(imgs, n_facies:int=3):
    """
    Convert images to facies‐label maps.
    Parameters:
    - imgs: numpy array of images (N, H, W) or (N, H, W, C)
    - n_facies: number of facies classes
    Returns:
    - facies_models: numpy array of shape (N, H, W) with integer labels 0..n_facies-1
    """
    facies_models = []
    for img in imgs:
        # flatten & compute bin edges
        _, bins = np.histogram(img.flatten(), bins=n_facies)
        labels = np.zeros_like(img, dtype=np.int32)
        for i in range(n_facies):
            if i < n_facies - 1:
                mask = (img >= bins[i]) & (img <  bins[i+1])
            else:
                mask = (img >= bins[i]) & (img <= bins[i+1])
            labels[mask] = i
        facies_models.append(labels)
    return np.stack(facies_models, axis=0)
        
    

def show_grid(batch, title, n_samples=None, n_cols=5,**kwargs):
    total = len(batch) if n_samples is None else min(len(batch), n_samples)
    n_rows = math.ceil(total / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*3, n_rows*3),
                             squeeze=False)
    fig.suptitle(title)
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        if idx < total:
            img = batch[idx, 0]
            ax.imshow(img,**kwargs)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
