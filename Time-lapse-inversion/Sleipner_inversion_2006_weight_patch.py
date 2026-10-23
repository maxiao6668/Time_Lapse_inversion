import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import scipy
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp

from matplotlib.colors import ListedColormap

from matplotlib import font_manager
from torch import nn

import tensorly as tl
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import pytorch_lightning as pl

import datetime as dt
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import segyio
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import segyio
from typing import Dict
import blobfile as bf
import wandb
from scipy.ndimage import gaussian_filter
from net_test import UNO3D_cai
from patch_blend_utils import make_weight_window, stitch_weighted_3d, finalize_weighted
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")   

cm_rtm=np.load("/home/max0b/no_vmb/colorbar_image.npy")
cm_rtm = ListedColormap(cm_rtm)
cm = ListedColormap(np.fromfile("/home/max0b/no_vmb/colorbar_ind (1).r@", dtype=np.float32).reshape(3,256).T)

def plot_slices(array, ids=None, size=5, width=1, ax=None, fig=None, recs=None, sous=None, fig_name=None, cbar_label='km/s', horizons=None, bounds=None, **kwargs):
    """
    Generate image slices from a 3D array.
    """
    if ids is None:
        ids = [array.shape[2]//2, array.shape[1]//2, array.shape[0]//2]

    if bounds is None:
        x = np.arange(array.shape[2])
        y = np.arange(array.shape[1])
        z = np.arange(array.shape[0])
    else:
        x = np.linspace(bounds[0], bounds[1], array.shape[2])
        y = np.linspace(bounds[0], bounds[2], array.shape[1])
        z = np.linspace(bounds[0], bounds[3], array.shape[0])
   
    
    grids = [x,y,z]

    if fig is None:
        fig = plt.figure(figsize=(1.5*size, size))
        
    if ax:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(size, size, subplot_spec=ax, wspace=0.05, hspace=0.05)
    else:
        gs = matplotlib.gridspec.GridSpec(size, size, wspace=0.3, hspace=0.3)

    if fig is None:
        ax = fig.add_subplot(gs[0:(size-width), 0:(size-width)])
    
    # Upper left image
    ax0 = fig.add_subplot(gs[0:(size-width-2), 0:(size-width-2)])

    if bounds is None:
        im0 = ax0.imshow(array[ids[2]], **kwargs)
    else:
        im0 = ax0.imshow(array[ids[2]], extent=[bounds[0], bounds[1], bounds[2], bounds[0]], **kwargs)
    if horizons is not None:
        ax0.imshow(horizons[ids[2]], cmap='jet')
    ax0.hlines(grids[1][ids[1]],xmin=grids[0].min(),xmax=grids[0].max(), linestyles='--', color='black')
    ax0.text(grids[0].max(), grids[1][ids[1]], 'X1', horizontalalignment='right')
    ax0.text(grids[0].min(), grids[1][ids[1]], 'X0')
    ax0.text(grids[0][ids[0]], grids[1].min(), 'Y0', verticalalignment='top')
    ax0.text(grids[0][ids[0]], grids[1].max(), 'Y1')
    ax0.vlines(grids[0][ids[0]],ymin=grids[1].min(),ymax=grids[1].max(), linestyles='--', color='black')
    
    if bounds is not None:
        ax0.set_xlabel('km')
        ax0.set_ylabel('km')
    else:
        ax0.set_xlabel('nx')
        ax0.set_ylabel('ny')
    ax0.xaxis.set_ticks_position('top')
    ax0.xaxis.set_label_position('top')

    # Right image
    ax1 = fig.add_subplot(gs[0:(size-width-2), (size-width-2):])
    if bounds is None:
        ax1.imshow(array[:, :, ids[0]].T, **kwargs)
    else:
        ax1.imshow(array[:, :, ids[0]].T, extent=[bounds[0], bounds[3], bounds[1], bounds[0]], **kwargs)
    if horizons is not None:
        ax1.imshow(horizons[:, :, ids[0]], cmap='jet')
    ax1.hlines(grids[1][ids[1]],xmin=grids[2].min(),xmax=grids[2].max(), linestyles='--', color='black')
    ax1.vlines(grids[2][ids[2]],ymin=grids[1].min(),ymax=grids[1].max(), linestyles='--', color='black')
    ax1.text(grids[2][ids[2]], grids[1].max(), 'Y1', horizontalalignment='left', verticalalignment='bottom')
    ax1.text(grids[2][ids[2]], grids[1].min(), 'Y0', verticalalignment='top')
    ax1.yaxis.set_ticks_position('right')
    # ax1.xaxis.set_ticks_position('top')
    
    # Bottom image
    ax2 = fig.add_subplot(gs[(size-width-2):, 0:(size-width-2)])
    
    if bounds is None:
        ax2.imshow(array[:, ids[1], :], **kwargs)
    else:
        ax2.imshow(array[:, ids[1], :], extent=[bounds[0], bounds[1], bounds[3], bounds[0]], **kwargs)
    if horizons is not None:
        ax2.imshow(horizons[:, ids[1], :], cmap='jet')
    ax2.hlines(grids[2][ids[2]],xmin=grids[0].min(),xmax=grids[0].max(), linestyles='--', color='black')
    ax2.vlines(grids[0][ids[0]],ymin=grids[2].min(),ymax=grids[2].max(), linestyles='--', color='black')
    ax2.text(grids[0].max(), grids[2][ids[2]], 'X1', horizontalalignment='right', verticalalignment='top')
    ax2.text(grids[0].min(), grids[2][ids[2]], 'X0', verticalalignment='top')
    if bounds is not None:
        ax2.set_ylabel('km')
    else:
        ax2.set_ylabel('nz')
    

    # Add colorbar for the first image
    cbar = fig.colorbar(im0, ax=[ax0, ax1, ax2], orientation='vertical', fraction=0.05, pad=0.075)
    cbar.set_label(cbar_label, rotation=270-180, labelpad=15)
    
    if recs is not None:
        ax0.scatter(recs[0], recs[1], c='yellow', marker='v')
        
    if sous is not None:
        ax0.scatter(sous[0], sous[1], c='black', marker='*')
        
    if fig_name is not None:
        plt.savefig(fig_name)
        
    plt.show()


def tv_loss_3d(model: torch.Tensor) -> torch.Tensor:
    """
    Isotropic TV for 3D volume.
    Penalizes differences along depth (z), height (y), and width (x).
    """
    d, h, w = model.shape

    # differences along each axis (crop to align shapes)
    dz = model[:d-1, :h-1, :w-1] - model[1:,   :h-1, :w-1]
    dy = model[:d-1, :h-1, :w-1] - model[:d-1, 1:,   :w-1]
    dx = model[:d-1, :h-1, :w-1] - model[:d-1, :h-1, 1:]

    # isotropic TV: sqrt(dx^2 + dy^2 + dz^2)
    tv = torch.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + 1e-15)

    return tv.sum() / (d * h * w)

cm_rtm=np.load("/home/max0b/no_vmb/colorbar_image.npy")
cm_rtm = ListedColormap(cm_rtm)
cm = ListedColormap(np.fromfile("/home/max0b/no_vmb/colorbar_ind (1).r@", dtype=np.float32).reshape(3,256).T)

# 1. register font file
font_path = '/home/max0b/no_rwi/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)

# 2. set global font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # also set sans-serif if needed

# (optional) make sure the minus sign renders correctly
plt.rcParams['axes.unicode_minus'] = False

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
    
model = UNO3D_cai(in_width=2+3, width=12).to(device)
ckpt_path = "/home/max0b/IP_inversion_NO/test_inversion1/test_inversion1-2025-12-07-15-41-25/checkpoints/epoch=22-step=20309.ckpt"#
checkpoint = torch.load(ckpt_path,map_location='cpu')
# extract the model state_dict
state_dict = checkpoint['state_dict']


# #model = FNO2d(mode_x, mode_y, width, input_channel, output_channel)
# #model.load_state_dict(torch.load('../data/model_fno_curveA.pth'))
model.load_state_dict(state_dict)
model.eval()

# Load seismic data
segy_file="/home/max0b/IP_inversion_NO/data/06p07ful.sgy"
with segyio.open(segy_file) as f:
    d_ = segyio.cube(f).transpose(2, 1, 0)[:]
    xlines = f.xlines if hasattr(f, 'xlines') else np.arange(d_.shape[2])
    ilines = f.ilines if hasattr(f, 'ilines') else np.arange(d_.shape[1])
    samples = f.samples[:] if hasattr(f, 'samples') else np.arange(d_.shape[0])

seismic_data=d_*180
seismic_data=seismic_data[300:600,0:300,:]
Back_model=np.load("/home/max0b/IP_inversion_NO/4DProximal/data/Sleipner/Sleipner_impedance_model.npy")
Back_model_3d = np.repeat(Back_model[:, :, None], 249, axis=2)[300:600,0:300,:]

model_network=model.to(device)
model_network.eval()
# ------------ patch configuration ------------
pd, ph = 100, 100         # D/H size of each patch; W uses the full length
overlap_d, overlap_h = 20, 20
stride_d, stride_h = pd - overlap_d, ph - overlap_h
gain, lr = 3, 50
n_steps = 100

D, H, W = Back_model_3d.shape
accum = np.zeros((D, H, W), dtype=np.float32)
weight_sum = np.zeros((D, H, W), dtype=np.float32)
weight_win = make_weight_window(pd, ph, window="hann")

def run_single_patch(bg_patch_np, data_patch_np):
    """Run inversion on a single patch; returns a numpy array of shape (d,h,w)."""
    init_v_np = scipy.ndimage.gaussian_filter(bg_patch_np, sigma=1)
    init_v = torch.tensor(init_v_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    init_v.requires_grad_(True)  # leaf tensor

    input_1 = torch.tensor(bg_patch_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    labels = torch.tensor(data_patch_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) * gain

    optim = torch.optim.Adam([init_v], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.99)
    mseloss = torch.nn.MSELoss()
    maeloss = torch.nn.L1Loss()

    for i in range(n_steps):
        inputs_no = torch.cat([input_1, init_v], dim=1)
        
        preds = model_network(inputs_no.permute(0, 2, 3, 4, 1).contiguous())

        loss = maeloss(preds, labels) * 0 + mseloss(preds, labels) * 1 + tv_loss_3d(init_v[0, 0]) * 5
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        # smooth the gradient
        with torch.no_grad():
            if init_v.grad is not None:
                grad_np = init_v.grad.squeeze().cpu().numpy()
                grad_smoothed = scipy.ndimage.gaussian_filter(grad_np, sigma=0)
                init_v.grad.zero_()
                init_v.grad.copy_(torch.tensor(grad_smoothed, device=device).unsqueeze(0).unsqueeze(0))

        if i % 5 == 0:
            print(f'patch step {i}, loss {loss.item()}')

    return init_v.detach().cpu().squeeze().numpy()

# ------------ iterate over all patches ------------
from tqdm import tqdm  # already imported at the top; redundant import is harmless

# ------------ iterate over all patches ------------
d_ranges = list(range(0, D, stride_d))
h_ranges = list(range(0, H, stride_h))

for d0 in tqdm(d_ranges, desc="Depth patches"):
    d1 = min(d0 + pd, D)
    for h0 in tqdm(h_ranges, desc=f"Height patches @d={d0}-{d1}", leave=False):
        h1 = min(h0 + ph, H)

        bg_patch = Back_model_3d[d0:d1, h0:h1, :]
        data_patch = seismic_data[d0:d1, h0:h1, :]

        out_patch = run_single_patch(bg_patch, data_patch)
        stitch_weighted_3d(out_patch, accum, weight_sum, d0, h0, weight_win)

        del bg_patch, data_patch, out_patch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
result_vol = finalize_weighted(accum, weight_sum)
np.save("/home/max0b/IP_inversion_NO/Paper/result_Sleipner_2006_weight_patch.npy", result_vol)
# result_vol is the full-size (1001, 468, 249) inversion result (on CPU)