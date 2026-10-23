#!/usr/bin/env python3
# ============================================================
# Sleipner_DH_patch_multi_gpu.py
# Multi-GPU patch inversion over (D,H) while keeping W full.
#
# - torchrun: torchrun --nproc_per_node=NGPUS Sleipner_DH_patch_multi_gpu.py ...
# - SLURM:   srun --ntasks=NGPUS --gpus-per-task=1 python Sleipner_DH_patch_multi_gpu.py ...
#
# Output:
#   - out_dir/partials/partial_rank{r}.npz
#   - out_dir/result_*.npy (rank0 merged)
# ============================================================

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import segyio
import scipy

from tqdm import tqdm

# ---- your utils ----
from patch_blend_utils import make_weight_window
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import scipy
import matplotlib.pyplot as plt

from torch import nn


import pytorch_lightning as pl

import datetime as dt
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import segyio

import numpy as np

from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import segyio



from net_test import UNO3D_cai
# ============================================================
# 1) DDP init (torchrun + SLURM)
# ============================================================
def init_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ============================================================
# 2) TV loss (keep your original if already defined)
# ============================================================
def tv_loss_3d(model: torch.Tensor) -> torch.Tensor:
    d, h, w = model.shape
    dz = model[:d-1, :h-1, :w-1] - model[1:,   :h-1, :w-1]
    dy = model[:d-1, :h-1, :w-1] - model[:d-1, 1:,   :w-1]
    dx = model[:d-1, :h-1, :w-1] - model[:d-1, :h-1, 1:]
    tv = torch.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + 1e-15)
    return tv.sum() / (d * h * w)

# ============================================================
# 3) Model: UNO3D_cai (import your existing implementation directly)
#    Placeholder: paste your existing UNO3D_cai implementation here
# ============================================================
class OperatorBlock_3D(nn.Module):
    """
    Combines spectral convolution and pointwise convolution.
    """
    def __init__(self, in_codim, out_codim, dim1=None, dim2=None, dim3=None,
                 modes1=16, modes2=16, modes3=16, Normalize=True, Non_Lin=True):
        super(OperatorBlock_3D, self).__init__()

        self.conv = SpectralConv3d_Uno(in_codim, out_codim,
                                       dim1, dim2, dim3,
                                       modes1, modes2, modes3)
        self.w = pointwise_op_3D(in_codim, out_codim)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.norm = nn.InstanceNorm3d(int(out_codim), affine=True)

    def forward(self, x):
        # automatically infer input size
        x1 = self.conv(x)
        x2 = self.w(x)
        x_out = x1 + x2

        if self.normalize:
            x_out = self.norm(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class SpectralConv3d_Uno(nn.Module):
    """
    3D Fourier layer supporting arbitrary input sizes.
    Performs FFT -> low-frequency filtering -> IFFT.
    """
    def __init__(self, in_codim, out_codim,
                 dim1=None, dim2=None, dim3=None,
                 modes1=16, modes2=16, modes3=16):
        super(SpectralConv3d_Uno, self).__init__()

        self.in_channels = int(in_codim)
        self.out_channels = int(out_codim)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.modes3 = int(modes3)

        # initialize complex weights (real, imag parts)
        scale = (1 / (2 * in_codim)) ** 0.5
        self.weights1 = nn.Parameter(scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, 2))

    def compl_mul3d(self, input, weights):
        """Complex multiplication in Fourier domain"""
        weights = torch.view_as_complex(weights)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """
        input: (B, Cin, D, H, W)
        output: (B, Cout, D, H, W)
        """
        B, Cin, D, H, W = x.shape
        # FFT
        x_ft = torch.fft.rfftn(x, s=(D, H, W), norm='forward')

        # clamp modes to at most half of the input size
        m1 = min(self.modes1, D // 2)
        m2 = min(self.modes2, H // 2)
        m3 = min(self.modes3, W // 2)

        # initialize output spectrum
        out_ft = torch.zeros(B, self.out_channels, D, H, W // 2 + 1,
                             device=x.device, dtype=torch.cfloat)

        # four spectral quadrants
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1[:, :, :m1, :m2, :m3, :])
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights2[:, :, :m1, :m2, :m3, :])
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.weights3[:, :, :m1, :m2, :m3, :])
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights4[:, :, :m1, :m2, :m3, :])

        # IFFT back to physical space
        x = torch.fft.irfftn(out_ft, s=(D, H, W), norm='forward')
        return x
import torch.nn.functional as F


class pointwise_op_3D(nn.Module):
    """
    Pointwise (1x1x1) convolution with optional spectral alias handling.
    Works for arbitrary input sizes.
    """
    def __init__(self, in_codim, out_codim):
        super(pointwise_op_3D, self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)

    def forward(self, x):
        return self.conv(x)
class UNO3D_cai(pl.LightningModule):
    """
    UNO3D variant supporting variable input sizes with fixed Fourier modes.
    """
    def __init__(self, in_width, width, pad=0, factor=3/4, pad_both=False):
        super(UNO3D_cai, self).__init__()
        self.in_width = in_width
        self.width = width
        self.padding = pad
        self.pad_both = pad_both
        self.loss_fn = torch.nn.MSELoss()
        # input projection
        self.fc_n1 = nn.Linear(in_width, width // 2)
        self.fc0 = nn.Linear(width // 2, width)

        # fixed number of Fourier modes
        # fixed number of Fourier modes
        # modes_conv0 = (96, 48, 48)
        # modes_conv1 = (64, 32, 32)
        # modes_conv2 = (32, 16, 16)
        # modes_conv6 = (32, 16, 16)
        # modes_conv7 = (64, 32, 32)
        # modes_conv8 = (96, 48, 48)
        
        modes_conv0 = (64, 48, 48)
        modes_conv1 = (50, 25, 25)
        modes_conv2 = (32, 16, 16)
        modes_conv6 = (32, 16, 16)
        modes_conv7 = (50, 25, 25)
        modes_conv8 = (64, 48, 48)       
        
        # modes_conv0 = (48, 24, 24)
        # modes_conv1 = (32, 16, 16)
        # modes_conv2 = (16, 8, 8)
        # modes_conv6 = (16, 8, 8)
        # modes_conv7 = (32, 16, 16)
        # modes_conv8 = (48, 24, 24)

        # build layer stack
        self.conv0 = OperatorBlock_3D(width, int(2 * factor * width), None, None, None, *modes_conv0)
        self.conv1 = OperatorBlock_3D(int(2 * factor * width), int(4 * factor * width), None, None, None, *modes_conv1)
        self.conv2 = OperatorBlock_3D(int(4 * factor * width), int(8 * factor * width), None, None, None, *modes_conv2)
        self.conv6 = OperatorBlock_3D(int(8 * factor * width), int(4 * factor * width), None, None, None, *modes_conv6)
        self.conv7 = OperatorBlock_3D(int(8 * factor * width), int(2 * factor * width), None, None, None, *modes_conv7)
        self.conv8 = OperatorBlock_3D(int(4 * factor * width), int(2 * width), None, None, None, *modes_conv8)

        # output layers
        self.fc1 = nn.Linear(3 * width, 4 * width)
        self.fc2 = nn.Linear(4 * width, 1)

    def forward(self, x):
        # append grid coordinates (x,y,z)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        # input projection
        x_fc = F.gelu(self.fc_n1(x))
        x_fc0 = F.gelu(self.fc0(x_fc))
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)

        # backbone
        x_c0 = self.conv0(x_fc0)
        x_c1 = self.conv1(x_c0)
        x_c2 = self.conv2(x_c1)

        x_c6 = self.conv6(x_c2)
        x_c6 = torch.cat([x_c6, x_c1], dim=1)

        x_c7 = self.conv7(x_c6)
        x_c7 = torch.cat([x_c7, x_c0], dim=1)

        x_c8 = self.conv8(x_c7)
        x_c8 = torch.cat([x_c8, x_fc0], dim=1)

        # Padding
        if self.padding != 0:
            if self.pad_both:
                x_c8 = x_c8[..., self.padding // 2:-self.padding // 2]
            else:
                x_c8 = x_c8[..., :-self.padding]

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)
        x_fc1 = F.gelu(self.fc1(x_c8))
        x_out = self.fc2(x_fc1)
        return x_out.permute(0, 4, 1, 2, 3)

    def get_grid(self, shape, device):
        """
        shape: (B, D, H, W, C)
        """
        B, D, H, W = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, D, device=device).view(1, D, 1, 1, 1).repeat(B, 1, H, W, 1)
        gridy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1, 1).repeat(B, D, 1, W, 1)
        gridz = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W, 1).repeat(B, D, H, 1, 1)
        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        x=x.permute(0, 2, 3, 4, 1).contiguous()
        #print(f"x_shape:{x.shape}")
        y_hat = self(x)
   

        data_loss = self.loss_fn(y_hat,y)
        eculedian_loss=F.pairwise_distance(y_hat, y, keepdim = True).mean()        
        loss = eculedian_loss +data_loss
        
        
        self.log("train_loss", loss, on_step=True, on_epoch=True,sync_dist=True)       
        
        self.log("eculedian_loss", eculedian_loss, on_step=True, on_epoch=True,sync_dist=True)    
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        x=x.permute(0, 2, 3, 4, 1).contiguous()
        # x, y = batch['x'], batch['y']
        y_hat = self(x)
        #loss_mse = self.loss_fn(y, y_hat)
        
        data_loss = self.loss_fn(y_hat,y)
        eculedian_loss=F.pairwise_distance(y_hat, y, keepdim = True).mean()        
        loss = eculedian_loss +data_loss
        
        
        self.log("val_loss", loss, on_step=True, on_epoch=True,sync_dist=True)       
        

    def configure_optimizers(self,lr=1e-3, step_size=50, gamma=0.7):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # or 'step' for per-step updates
                       'frequency': 1}
        return [optimizer], [scheduler]

# ============================================================
# 4) Patch tasks: patch over (D,H), keep W full
# ============================================================
def build_tasks(D: int, H: int, pd: int, ph: int, stride_d: int, stride_h: int) -> List[Tuple[int,int,int,int]]:
    tasks = []
    for d0 in range(0, D, stride_d):
        d1 = min(d0 + pd, D)
        for h0 in range(0, H, stride_h):
            h1 = min(h0 + ph, H)
            tasks.append((d0, d1, h0, h1))
    return tasks

def stitch_weighted_dh(
    patch_3d: np.ndarray,      # (pd', ph', W)
    accum: np.ndarray,         # (D, H, W)
    weight_sum: np.ndarray,    # (D, H, W)
    d0: int,
    h0: int,
    weight_win_dh: np.ndarray  # (pd, ph)
) -> None:
    pd2, ph2, wlen = patch_3d.shape
    win = weight_win_dh[:pd2, :ph2][:, :, None]  # -> (pd2, ph2, 1) broadcast on W
    accum[d0:d0+pd2, h0:h0+ph2, :] += patch_3d * win
    weight_sum[d0:d0+pd2, h0:h0+ph2, :] += win

def finalize_weighted(accum: np.ndarray, weight_sum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return accum / (weight_sum + eps)

# ============================================================
# 5) Single patch inversion (matches the original logic)
# ============================================================

@torch.no_grad()
def _to_tensor(x_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x_np, device=device, dtype=torch.float32)

def run_single_patch(
    bg_patch_np: np.ndarray,     # (pd', ph', W)
    data_patch_np: np.ndarray,   # (pd', ph', W)
    model: nn.Module,
    device: torch.device,
    gain: float,
    lr: float,
    n_steps: int,
    
    grad_sigma: float,
    w_mse: float,
    w_mae: float,
    w_tv: float,
    step_size: int,
    gamma: float,
    print_every: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    # trainable init_v
    """
    Patch inversion optimizing init_v to match observed data.
    bg_patch_np:   (D, ph, pw)
    data_patch_np: (D, ph, pw)
    returns:
        inv_patch:  (D, ph, pw)  # inverted acoustic impedance
        pred_patch: (D, ph, pw)  # final predicted seismic (after inversion)
    """
    # Init model (trainable)
    init_v_np = scipy.ndimage.gaussian_filter(bg_patch_np, sigma=0)
    init_v = torch.tensor(init_v_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    init_v.requires_grad_(True)  # leaf tensor
    
    
    input_1 = _to_tensor(bg_patch_np, device).unsqueeze(0).unsqueeze(0)
    labels = _to_tensor(data_patch_np, device).unsqueeze(0).unsqueeze(0) * gain

    optim = torch.optim.Adam([init_v], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    mseloss = torch.nn.MSELoss()
    maeloss = torch.nn.L1Loss()

    final_preds = None  # will store last-iteration prediction

    for i in range(n_steps):
        inputs_no = torch.cat([input_1, init_v], dim=1)  # (1,2,D,H,W)

        preds = model(inputs_no.permute(0, 2, 3, 4, 1).contiguous())  # (1,1,D,H,W)

        # Save final prediction (last iteration)
        if i == n_steps - 1:
            final_preds = preds.detach().cpu().squeeze().numpy()

        loss = (w_mae * maeloss(preds, labels) +
                w_mse * mseloss(preds, labels) +
                w_tv  * tv_loss_3d(init_v[0, 0]))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        scheduler.step()

        # Smooth gradients on init_v
        with torch.no_grad():
            if init_v.grad is not None and grad_sigma > 0:
                grad_np = init_v.grad.squeeze().detach().cpu().numpy()
                grad_smoothed = scipy.ndimage.gaussian_filter(grad_np, sigma=grad_sigma)
                init_v.grad.zero_()
                init_v.grad.copy_(torch.tensor(grad_smoothed, device=device).unsqueeze(0).unsqueeze(0))

        if print_every and (i % print_every == 0):
            print(f"[patch] step {i:04d} loss={float(loss.item()):.6e}")

    inv_patch = init_v.detach().cpu().squeeze().numpy()
    return inv_patch, final_preds

# ============================================================
# 6) I/O: load segy + build background 3D
# ============================================================
def load_segy_cube(segy_file: str) -> np.ndarray:
    with segyio.open(segy_file) as f:
        d_ = segyio.cube(f).transpose(2, 1, 0)[:]  # (T, IL, XL) style after transpose -> check your original
    return d_.astype(np.float32)

# ============================================================
# 7) Main
# ============================================================
def main():
    
    parser = argparse.ArgumentParser()

    # --- model ---
    parser.add_argument("--ckpt_path", type=str, required=True, help="Lightning .ckpt or state_dict .pth")
    parser.add_argument("--in_width", type=int, default=5)
    parser.add_argument("--width", type=int, default=12)

    # --- data ---
    parser.add_argument("--segy_file", type=str, required=True)
    parser.add_argument("--back_model_npy", type=str, required=True, help="2D background model (D0,H0) npy")
    parser.add_argument("--scale_seis", type=float, default=180.0, help="Multiply seismic by this factor")

    # --- crop (match your Sleipner slice) ---

    parser.add_argument("--w_len", type=int, default=249, help="Repeat background along W and crop to this length")

    # --- patch config over (D,H) ---
    parser.add_argument("--pd", type=int, default=100)
    parser.add_argument("--ph", type=int, default=100)
    parser.add_argument("--overlap_d", type=int, default=20)
    parser.add_argument("--overlap_h", type=int, default=20)
    parser.add_argument("--window", type=str, default="hann")

    # --- inversion hyperparams ---
    parser.add_argument("--gain", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=50.0)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--init_sigma", type=float, default=1.0)
    parser.add_argument("--grad_sigma", type=float, default=0.0)
    parser.add_argument("--w_mse", type=float, default=1.0)
    parser.add_argument("--w_mae", type=float, default=0.0)
    parser.add_argument("--w_tv", type=float, default=5.0)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--print_every", type=int, default=0)

    # --- output ---
    parser.add_argument("--out_dir", type=str, default="./out_sleipner_multi_gpu")
    parser.add_argument("--final_name", type=str, default="result_Sleipner_inverted.npy")
    parser.add_argument("--final_pred_name", type=str, default="result_Sleipner_pred_seis.npy")

    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    partial_dir = os.path.join(args.out_dir, "partials")
    os.makedirs(partial_dir, exist_ok=True)

    # ---- load model ----
    model = UNO3D_cai(in_width=args.in_width, width=args.width).to(device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        # if it's a raw state_dict
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    # ---- load data (each rank loads; simplest & consistent with your template) ----
    seismic = load_segy_cube(args.segy_file) * args.scale_seis
    seismic = seismic # (D,H,Wfull from segy)
    # the original script does not crop W; just make sure back_model W matches seismic
    # if the SEGY W != w_len, set args.w_len to the real W or crop/align seismic accordingly
    W = seismic.shape[2]

    Back_model_2d = np.load(args.back_model_npy).astype(np.float32)
    Back_model_3d = np.repeat(Back_model_2d[:, :, None], args.w_len, axis=2).astype(np.float32)
    Back_model_3d = Back_model_3d

    # align W
    if Back_model_3d.shape[2] != W:
        Wmin = min(Back_model_3d.shape[2], W)
        seismic = seismic[:, :, :Wmin]
        Back_model_3d = Back_model_3d[:, :, :Wmin]

    D, H, W = Back_model_3d.shape

    # ---- patch config ----
    stride_d = args.pd - args.overlap_d
    stride_h = args.ph - args.overlap_h
    if stride_d <= 0 or stride_h <= 0:
        raise ValueError("Stride must be positive. Ensure overlap < patch size.")

    weight_win_dh = make_weight_window(args.pd, args.ph, window=args.window).astype(np.float32)

    # per-rank accumulators
    accum_rank = np.zeros((D, H, W), dtype=np.float32)
    weight_sum_rank = np.zeros((D, H, W), dtype=np.float32)

    accum_pred_rank = np.zeros((D, H, W), dtype=np.float32)
    weight_pred_rank = np.zeros((D, H, W), dtype=np.float32)

    # ---- tasks split ----
    tasks = build_tasks(D, H, args.pd, args.ph, stride_d, stride_h)
    my_tasks = tasks[rank::world_size]

    pbar = tqdm(my_tasks, desc=f"Rank {rank}/{world_size} patches", disable=(rank != 0))

    for (d0, d1, h0, h1) in pbar:
        bg_patch = Back_model_3d[d0:d1, h0:h1, :]     # (pd', ph', W)
        data_patch = seismic[d0:d1, h0:h1, :]         # (pd', ph', W)

        inv_patch, pred_patch = run_single_patch(
            bg_patch_np=bg_patch,
            data_patch_np=data_patch,
            model=model,
            device=device,
            gain=args.gain,
            lr=args.lr,
            n_steps=args.n_steps,
            grad_sigma=args.grad_sigma,
            w_mse=args.w_mse,
            w_mae=args.w_mae,
            w_tv=args.w_tv,
            step_size=args.step_size,
            gamma=args.gamma,
            print_every=args.print_every
        )

        stitch_weighted_dh(inv_patch, accum_rank, weight_sum_rank, d0, h0, weight_win_dh)
        stitch_weighted_dh(pred_patch, accum_pred_rank, weight_pred_rank, d0, h0, weight_win_dh)

        del bg_patch, data_patch, inv_patch, pred_patch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- save partial ----
    partial_path = os.path.join(partial_dir, f"partial_rank{rank}.npz")
    np.savez_compressed(
        partial_path,
        accum=accum_rank,
        weight_sum=weight_sum_rank,
        accum_pred=accum_pred_rank,
        weight_pred=weight_pred_rank,
        shape=np.array([D, H, W], dtype=np.int64)
    )

    ddp_barrier()

    # ---- merge on rank0 ----
    if rank == 0:
        accum = np.zeros((D, H, W), dtype=np.float32)
        weight_sum = np.zeros((D, H, W), dtype=np.float32)

        accum_pred = np.zeros((D, H, W), dtype=np.float32)
        weight_pred = np.zeros((D, H, W), dtype=np.float32)

        for r in range(world_size):
            p = np.load(os.path.join(partial_dir, f"partial_rank{r}.npz"))
            accum += p["accum"]
            weight_sum += p["weight_sum"]
            accum_pred += p["accum_pred"]
            weight_pred += p["weight_pred"]

        result_vol = finalize_weighted(accum, weight_sum)
        pred_vol = finalize_weighted(accum_pred, weight_pred)

        out_inv = os.path.join(args.out_dir, args.final_name)
        out_pred = os.path.join(args.out_dir, args.final_pred_name)

        np.save(out_inv, result_vol.astype(np.float32))
        np.save(out_pred, pred_vol.astype(np.float32))

        print(f"[rank0] Saved merged inverted volume: {out_inv}")
        print(f"[rank0] Saved merged predicted seismic: {out_pred}")

    ddp_barrier()
    ddp_cleanup()

if __name__ == "__main__":
    main()
