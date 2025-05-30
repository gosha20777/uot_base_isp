# from src.train import train
# from src.configs.config import TaskConfig
import torch
import numpy as np
import argparse

import os, sys
# sys.path.append("..")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
import torch
import torch.nn as nn
import torchvision
import gc

# from src import distributions
import torch.nn.functional as F

from src.models.resnet import ResNet_D
from src.models.unet import UNet
from src.models.u2net import U2NET, U2NETP, EMA

from src.dataset.dataset import unfreeze, freeze, weights_init_D, load_dataset, load_celeba
from src.metrics.fid_score import calculate_frechet_distance
from src.metrics.inception import InceptionV3
# from src.plotters import plot_random_images, plot_images

from copy import deepcopy
import json

from tqdm import tqdm # tqdm_notebook as tqdm
from IPython.display import clear_output

import wandb
from src.dataset.dataset import fig2data, fig2img # for wandb

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


class Args(argparse.Namespace):
    image_size=256
    centered=True
    num_channels_dae=64
    n_mlp=3
    ch_mult=[1,1,2,2,4,4]
    num_res_blocks=2
    attn_resolutions=(16,)
    dropout=0.
    resamp_with_conv=True
    conditional=True
    fir=True
    fir_kernel=[1, 3, 3, 1]
    skip_rescale=True
    resblock_type='biggan'
    progressive='none'
    progressive_input='residual'
    progressive_combine='sum'
    embedding_type='positional'
    fourier_scale=16.
    not_use_tanh=False
    z_emb_dim=256
    nz=100
    ngf=64
    num_channels=3
    batch_size=32
    

def get_pushed_loader_stats_old(T, loader, batch_size=8, verbose=False, device='cuda',
                            use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    freeze(model); freeze(T) 
  
    size = len(loader.dataset)

    pred_arr = np.empty((size, dims))

    with torch.no_grad():
        start_idx = 0
        for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                z = torch.randn(end-start, 100).to(device)
                batch = T(X[start:end].type(torch.FloatTensor).to(device), z).add(1).mul(.5)
                output = model(batch)[0].cpu().numpy().reshape(end-start, -1)
                pred_arr[start_idx: start_idx + output.shape[0]] = output 
                start_idx += output.shape[0]
    
   
    #pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def plot_images(X, Y, T):
    freeze(T);
    z = torch.randn(X.size(0), 100).cuda()
    with torch.no_grad():
        T_X = T(X, z)
        imgs = torch.cat([X, T_X, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X)', fontsize=24)
    axes[2, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes


def plot_random_images(X_sampler, Y_sampler, T):
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_images(X, Y, T)


if __name__ == "__main__":
    # download data in bash using commands:
    # bash ./datasets/scripts/download_hdf5_dataset.sh handbag_64
    # bash ./datasets/scripts/download_hdf5_dataset.sh shoes_64
    torch.manual_seed(0xBADBEEF)
    np.random.seed(0xBADBEEF)
    torch.cuda.manual_seed(0xBADBEEF)
    torch.backends.cudnn.deterministic = True
    
#     cfg = TaskConfig()
#     train(cfg)
    args=Args()
    
    DEVICE_IDS = [0, 1, 2, 3]

    # DATASET1, DATASET1_PATH = 'outdoor', '../pSp/datasets/outdoor_128.hdf5'
    # DATASET2, DATASET2_PATH = 'church', './datasets/church_128.hdf5'

    DATASET1, DATASET1_PATH = 'celeba_male', 'datasets/CelebA_HQ'
    DATASET2, DATASET2_PATH = 'celeba_female', 'datasets/CelebA_HQ'

    T_ITERS = 1
    f_LR, T_LR = 1e-4, 2e-4
    IMG_SIZE = 256

    BATCH_SIZE = 32

    PLOT_INTERVAL = 200
    COST = 'mse' # Mean Squared Error
    CPKT_INTERVAL = 1000
    MAX_STEPS = 100001
    SEED = 0x000000

    EXP_NAME = f'{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{IMG_SIZE}_uot_softplus_batch32'
    OUTPUT_PATH = '../checkpoints/{}/{}_{}_{}/'.format(COST, DATASET1, DATASET2, IMG_SIZE)
    
    config = dict(
        DATASET1=DATASET1,
        DATASET2=DATASET2, 
        T_ITERS=T_ITERS,
        f_LR=f_LR, T_LR=T_LR,
        BATCH_SIZE=BATCH_SIZE
    )

    assert torch.cuda.is_available()
    torch.cuda.set_device(f'cuda:{DEVICE_IDS[0]}')


    # datasets
    X_sampler = load_celeba('datasets/CelebA_HQ', "male", True, batch_size=BATCH_SIZE)
    X_test_sampler = load_celeba('datasets/CelebA_HQ', "male", False, batch_size=BATCH_SIZE)

    Y_sampler = load_celeba('datasets/CelebA_HQ', "female", True, batch_size=BATCH_SIZE)
    Y_test_sampler = load_celeba('datasets/CelebA_HQ', "female", False, batch_size=BATCH_SIZE)

    torch.cuda.empty_cache(); gc.collect()
    
    
    # loader stats
    filename = './{}_{}_test.json'.format(DATASET2, IMG_SIZE)
    with open(filename, 'r') as fp:
        data_stats = json.load(fp)
        mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
    del data_stats
    
    
    # models
    from src.models.ncsn.models.ncsnpp_generator_adagn import NCSNpp
    T = NCSNpp(args).cuda()
    # T = UNet(3, 3, base_factor=64).cuda()

    from src.models.ncsn.models.discriminator import Discriminator_large, Discriminator_small
    f = Discriminator_large(nc=args.num_channels, ngf=args.ngf, act=nn.LeakyReLU(0.2)).cuda()
    # f = ResNet_D(IMG_SIZE, nc=3).cuda()
    # f.apply(weights_init_D)

    print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
    print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]))

    
    mu, sigma = get_pushed_loader_stats_old(T, X_test_sampler.loader, verbose=False)
    print("starting fid calculation")
    fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
    print(fid)
    
    # ema
    import copy

    ema = EMA(0.999)
    ema_model = copy.deepcopy(T).cuda()

    start_ema = 30000
    
    # fixed samples
    X_fixed = X_sampler.sample(10)
    Y_fixed = Y_sampler.sample(10)
    X_test_fixed = X_test_sampler.sample(10)
    Y_test_fixed = Y_test_sampler.sample(10)

    fig, axes = plot_images(X_fixed, Y_fixed, T)
    plt.show(); plt.close(fig)
    
    # wandb
    os.system('wandb login <your_token>')
    
    wandb.init(name=EXP_NAME, project='progressive_growing_OT_first_gpu', config=config)
    pass

    # optimizers
    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, betas=(0.5, 0.9))
    f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, betas=(0.5, 0.9))

    sch_f = torch.optim.lr_scheduler.CosineAnnealingLR(f_opt, 700, eta_min=1e-5)
    sch_T = torch.optim.lr_scheduler.CosineAnnealingLR(T_opt, 700, eta_min=1e-5)
    
    # for UOT
    phi1 = lambda x: 2*F.softplus(x) - 2*F.softplus(0*x) # torch.exp(x)
    phi2 = lambda x: 2*F.softplus(x) - 2*F.softplus(0*x) # torch.exp(x)

    tau = 0.00001
    r1_gamma = 5  
    
    T = nn.DataParallel(T, device_ids=DEVICE_IDS)   
    f = nn.DataParallel(f, device_ids=DEVICE_IDS) 

    # main cycle
    for step in tqdm(range(MAX_STEPS)):
        # f optimization
        unfreeze(T); unfreeze(f)

        X = X_sampler.sample(BATCH_SIZE)
        z = torch.randn(BATCH_SIZE, args.nz).cuda()
        T_X = T(X, z)
        
        Y = Y_sampler.sample(BATCH_SIZE)
        Y.requires_grad = True
        f_opt.zero_grad()

        f_real = f(Y)
        f_loss_real = phi2(-f_real).mean()
        f_loss_real.backward(retain_graph=True)

        grad_ = torch.autograd.grad(outputs=f_real.sum(), inputs=Y, create_graph=True)[0]
        grad_penalty = (grad_.view(grad_.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = r1_gamma / 2 * grad_penalty
        grad_penalty.backward()


        f_loss_fake = phi1(f(T_X) - tau * torch.sum(((T_X - X).view(X.size(0), -1))**2, dim=1)).mean()
        f_loss_fake.backward()

        f_loss = f_loss_real + f_loss_fake

        f_opt.step()
        wandb.log({f'f_loss' : f_loss.item()}, step=step)
        del f_loss, Y, X, T_X, z; gc.collect(); torch.cuda.empty_cache()



        # T optimization
        freeze(f)
        
        T_opt.zero_grad()
        X = X_sampler.sample(BATCH_SIZE)
        z = torch.randn(BATCH_SIZE, args.nz).cuda()
        T_X = T(X, z)
        
        T_loss = (tau * torch.sum(((T_X - X).view(X.size(0), -1))**2, dim=1) - f(T_X)).mean()
        T_loss.backward(); T_opt.step()
        wandb.log({f'T_loss' : T_loss.item()}, step=step)
        del T_loss, T_X, X, z; gc.collect(); torch.cuda.empty_cache()

        if (step + 1) % 500 == 0:
            sch_f.step()
            sch_T.step()

        if (step + 1) == start_ema:
            ema_model = copy.deepcopy(T).cuda()
            freeze(ema_model)
        if (step + 1) > start_ema:
            ema.update_model_average(ema_model, T)

        if (step + 1) % PLOT_INTERVAL == 0:
            print('Plotting')
            clear_output(wait=True)

            fig, axes = plot_images(X_fixed, Y_fixed, T)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step+1) 
            plt.show(); plt.close(fig) 

            fig, axes = plot_random_images(X_sampler,  Y_sampler, T)
            wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step+1) 
            plt.show(); plt.close(fig) 

            fig, axes = plot_images(X_test_fixed, Y_test_fixed, T)
            wandb.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step+1) 
            plt.show(); plt.close(fig) 

            fig, axes = plot_random_images(X_test_sampler, Y_test_sampler, T)
            wandb.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step+1) 
            plt.show(); plt.close(fig)

            if (step + 1) >= start_ema:

                fig, axes = plot_images(X_fixed, Y_fixed, ema_model)
                wandb.log({'Fixed Images EMA' : [wandb.Image(fig2img(fig))]}, step=step+1) 
                plt.show(); plt.close(fig) 

                fig, axes = plot_random_images(X_sampler,  Y_sampler, ema_model)
                wandb.log({'Random Images EMA' : [wandb.Image(fig2img(fig))]}, step=step+1) 
                plt.show(); plt.close(fig) 


        if (step + 1) % CPKT_INTERVAL == 0:
            freeze(T);

            print('Computing FID')
            mu, sigma = get_pushed_loader_stats_old(T, X_test_sampler.loader)
            fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
            wandb.log({f'FID (Test)' : fid}, step=step+1)
            del mu, sigma


            if (step + 1) >= start_ema:
                mu, sigma = get_pushed_loader_stats_old(ema_model, X_test_sampler.loader)
                fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
                wandb.log({f'FID EMA (Test)' : fid}, step=step+1)
                del mu, sigma


    #         torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt'))
    #         torch.save(f.state_dict(), os.path.join(OUTPUT_PATH, f'f_{SEED}_{step}.pt'))
    #         torch.save(f_opt.state_dict(), os.path.join(OUTPUT_PATH, f'f_opt_{SEED}_{step}.pt'))
    #         torch.save(T_opt.state_dict(), os.path.join(OUTPUT_PATH, f'T_opt_{SEED}_{step}.pt'))

        gc.collect(); torch.cuda.empty_cache()
