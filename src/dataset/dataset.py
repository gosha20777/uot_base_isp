import pandas as pd
import numpy as np

import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from PIL import Image
from src.metrics.inception import InceptionV3
from tqdm import tqdm_notebook as tqdm
from src.metrics.fid_score import calculate_frechet_distance
from src.dataset.distributions import LoaderSampler
import h5py
from torch.utils.data import TensorDataset

import gc

from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import ImageFolder


def load_dataset(name, path, img_size=64, batch_size=64, test_ratio=0.1, device='cuda', encoder=None, is_latent=False):
    if name in ['shoes', 'handbag', 'outdoor', 'church']:
        dataset = h5py_to_dataset(path, name, img_size, encoder, device, is_latent)
    elif name in ['celeba_female', 'celeba_male', 'aligned_anime_faces', 'describable_textures']:
        transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = ImageFolder(path, transform=transform)
    else:
        raise Exception('Unknown dataset')
        
    if name in ['celeba_female', 'celeba_male']:
        with open('../datasets/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()[2:]
        if name == 'celeba_female':
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        else:
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] != '-1']
    elif dataset == 'describable_textures':
        idx = np.random.RandomState(seed=0xBADBEEF).permutation(len(dataset))
    else:
        idx = list(range(len(dataset)))
    
    
    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
    

    train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    test_sampler = LoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return train_sampler, test_sampler


def load_celeba(data_dir, gender, is_train=True, img_size=256, batch_size=64, device='cuda', encoder=None, is_latent=False):
    
    if not is_latent:
        
        path = os.path.join(data_dir, 'train' if is_train else 'test', gender)

        #transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if is_train:
            transform = Compose([Resize((img_size, img_size)), RandomHorizontalFlip(), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        dataset = ImageFolder(path, transform=transform)
        
        data_loader = DataLoader(dataset, shuffle=True, num_workers=8, batch_size=batch_size)
        
        encoded = []
        
        if not encoder:
            
            sampler = LoaderSampler(data_loader, device)
            return sampler
    
        with torch.no_grad():
            for X, _ in tqdm(data_loader):
                b_encoded = encoder.encode(X.type(torch.FloatTensor).to(device)).latent_dist.sample().mul(0.18215).cpu().data.numpy()
                encoded.extend(b_encoded)
                
        d1 = np.array(encoded)
        path_enc = os.path.join("datasets/CelebA_latent_vae", 'train' if is_train else 'test', gender, 'data_encoded32.h5')
        hf = h5py.File(path_enc, 'w')
        hf.create_dataset('dataset', data=d1)
        hf.close()
        

        sampler = LoaderSampler(data_loader, device)

        return sampler
    
    
    path = os.path.join(data_dir, 'train' if is_train else 'test', gender, "data_encoded32.h5")
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = torch.tensor(np.array(data), dtype=torch.float32)
#         dataset = F.interpolate(dataset, img_size, mode='bilinear')
    dataset = TensorDataset(dataset, torch.zeros(len(dataset)))
    sampler = LoaderSampler(DataLoader(dataset, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return sampler


def load_lsun(path, name="church_outdoor", batch_size=32, img_size=256, device='cuda'):
    data = datasets.LSUN(root=path, classes=[name+'_train'],
    transform=Compose([
        Scale(img_size),
        CenterCrop(img_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    test_data = datasets.LSUN(root=path, classes=[name+'_val'],
    transform=Compose([
        Scale(img_size),
        CenterCrop(img_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    sampler = LoaderSampler(DataLoader(data, batch_size, True, num_workers=8), device)
    test_sampler = LoaderSampler(DataLoader(test_data, batch_size, True, num_workers=8), device)
    return sampler, test_sampler


def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def h5py_to_dataset(path, name, img_size=64, encoder=None, device='cuda', is_latent=False):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        if not is_latent:
            dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        else:
            dataset = torch.tensor(np.array(data), dtype=torch.float32)
        dataset = F.interpolate(dataset, img_size, mode='bilinear')
        
#     dataset = dataset[:32*16]
        
    if encoder:
        encoded = []
        
        batch_size = 16
    
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size)):
                start, end = i, min(i + batch_size, len(dataset))
                batch = dataset[start:end]
                b_encoded = encoder.encode_first_stage(batch.type(torch.FloatTensor).to(device)).cpu().data.numpy()
                encoded.extend(b_encoded)
                
        d1 = np.array(encoded)
        hf = h5py.File(f'z_{name}/data_encoded.h5', 'w')
        hf.create_dataset('dataset', data=d1)
        hf.close()
        
        return TensorDataset(torch.tensor(np.array(encoded), dtype=torch.float32), torch.zeros(len(dataset)))
    
    return TensorDataset(dataset, torch.zeros(len(dataset)))

def get_loader_stats(loader, batch_size=8, device='cuda', verbose=False, use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model)
    
    size = len(loader.dataset)
    pred_arr = []
    
    with torch.no_grad():
        for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).to(device)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def get_pushed_loader_stats(T, alpha, loader, batch_size=8, verbose=False, device='cuda',
                            use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    with torch.no_grad():
        for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                batch = T(X[start:end].type(torch.FloatTensor).to(device), alpha).add(1).mul(.5)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def get_Z_pushed_loader_stats(T, loader, ZC=1, Z_STD=0.1, batch_size=8, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    with torch.no_grad():
        for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            Z = torch.randn(len(X), ZC, X.size(2), X.size(3)) * Z_STD
            XZ = torch.cat([X,Z], dim=1)
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                batch = T(XZ[start:end].type(torch.FloatTensor).to(device)).add(1).mul(.5)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma
