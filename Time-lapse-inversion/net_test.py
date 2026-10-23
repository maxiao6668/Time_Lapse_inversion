
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from collections import defaultdict
import sys, os
sys.path.append(os.path.join(os.getcwd(), "Neural_Solver_Library"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.nn import init
import warnings

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from torch.optim.swa_utils import AveragedModel, update_bn
#SWA
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from Neural_Solver_Library.models.U_FNO import Model
from types import SimpleNamespace

# ==========================================
# ========== Fourier Neural Layer ==========
# ==========================================

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


# ==========================================
# ========== Pointwise 1x1 Conv ============
# ==========================================

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


# ==========================================
# ========== Operator Block 3D =============
# ==========================================

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


# ==========================================
# ============ UNO3D model ==================
# ==========================================

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
        
        # modes_conv0 = (64, 48, 48)
        # modes_conv1 = (50, 25, 25)
        # modes_conv2 = (32, 16, 16)
        # modes_conv6 = (32, 16, 16)
        # modes_conv7 = (50, 25, 25)
        # modes_conv8 = (64, 48, 48)       
        
        modes_conv0 = (48, 24, 24)
        modes_conv1 = (32, 16, 16)
        modes_conv2 = (16, 8, 8)
        modes_conv6 = (16, 8, 8)
        modes_conv7 = (32, 16, 16)
        modes_conv8 = (48, 24, 24)
        
        # modes_conv0 = (24, 12, 12)
        # modes_conv1 = (16, 8, 8)
        # modes_conv2 = (8, 4, 4)
        # modes_conv6 = (8, 4, 4)
        # modes_conv7 = (16, 8, 8)
        # modes_conv8 = (24, 12, 12)

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
        eculedian_loss=F.pairwise_distance(y_hat, y).mean()        
        loss = eculedian_loss
        
        
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
        eculedian_loss=F.pairwise_distance(y_hat, y).mean()        
        loss = eculedian_loss
        
        
        self.log("val_loss", loss, on_step=True, on_epoch=True,sync_dist=True)       
        

    def configure_optimizers(self,lr=1e-3, step_size=50, gamma=0.7):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # or 'step' for per-step updates
                       'frequency': 1}
        return [optimizer], [scheduler]
    
    
    

class UNO3D_cai_mod(pl.LightningModule):
    """
    UNO3D variant supporting variable input sizes with fixed Fourier modes.
    """
    def __init__(self, in_width, width, pad=0, factor=3/4, pad_both=False):
        super(UNO3D_cai_mod, self).__init__()
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
        
        # modes_conv0 = (64, 48, 48)
        # modes_conv1 = (50, 25, 25)
        # modes_conv2 = (32, 16, 16)
        # modes_conv6 = (32, 16, 16)
        # modes_conv7 = (50, 25, 25)
        # modes_conv8 = (64, 48, 48)       
        
        modes_conv0 = (48, 24, 24)
        modes_conv1 = (32, 16, 16)
        modes_conv2 = (16, 8, 8)
        modes_conv6 = (16, 8, 8)
        modes_conv7 = (32, 16, 16)
        modes_conv8 = (48, 24, 24)
        
        # modes_conv0 = (24, 12, 12)
        # modes_conv1 = (16, 8, 8)
        # modes_conv2 = (8, 4, 4)
        # modes_conv6 = (8, 4, 4)
        # modes_conv7 = (16, 8, 8)
        # modes_conv8 = (24, 12, 12)

        # build layer stack
        self.conv0 = OperatorBlock_3D(width, int(2 * factor * width), None, None, None, *modes_conv0)
        self.conv1 = OperatorBlock_3D(int(2 * factor * width), int(4 * factor * width), None, None, None, *modes_conv1)
        self.conv2 = OperatorBlock_3D(int(4 * factor * width), int(8 * factor * width), None, None, None, *modes_conv2)
        self.conv6 = OperatorBlock_3D(int(8 * factor * width), int(4 * factor * width), None, None, None, *modes_conv6)
        self.conv7 = OperatorBlock_3D(int(8 * factor * width), int(2 * factor * width), None, None, None, *modes_conv7)
        self.conv8 = OperatorBlock_3D(int(4 * factor * width), int(2 * width), None, None, None, *modes_conv8)

        # output layers
        self.fc1 = nn.Linear(3 * width, 1 * width)
        self.fc2 = nn.Linear( 1* width, 1)

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
        eculedian_loss=F.pairwise_distance(y_hat, y).mean()
        loss = eculedian_loss


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
        eculedian_loss=F.pairwise_distance(y_hat, y).mean()
        loss = eculedian_loss


        self.log("val_loss", loss, on_step=True, on_epoch=True,sync_dist=True)


    def configure_optimizers(self,lr=1e-3, step_size=50, gamma=0.7):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # or 'step' for per-step updates
                       'frequency': 1}
        return [optimizer], [scheduler]
