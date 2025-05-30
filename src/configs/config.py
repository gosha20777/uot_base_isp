from dataclasses import dataclass, field
import torch
from typing import List


@dataclass
class TaskConfig:
    device: torch.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    batch_size: int = 64
    leaky: float = 0.2
    chs: int = 512
    num_blocks: int = 3

    show_every: int = 200
    save_every: int = 100000
    project_name: str = 'progressive_growing_OT_first_gpu'
    data_name_x: str = "outdoor"
    data_path_x: str = "../pSp/datasets/outdoor_128.hdf5"
    data_name_y: str = "church"
    data_path_y: str = "./datasets/church_128.hdf5"
    D_LR: float = 1e-4
    T_LR: float = 1e-4
    resolution: int = 5
    loss_tol: float = 0
    MAX_STEPS: int = 100000
    base_steps: int = 4000
    T_steps: int = 10
    validation: int = 0
    weight_decay: float = 1e-10
    base_factor: int = 48
    exp_name: str = "outdoor_church_32_inception_768"
    save_models: bool = False
    compute_fid_every: int = 1000
    full_log: int = 50
    latent_dim: int = 256
    resolutions_alae: List = field(default_factory=lambda: [4, 8, 16, 32, 64, 64])
    channels_alae: List = field(default_factory=lambda: [512, 256, 128, 128, 64, 32, 16])
    lambda_f: float = 1.0
    lambda_h: float = 1.0