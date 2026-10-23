#!/bin/bash
# =============================================================================
# Multi-GPU launcher for the full-scale Sleipner patch-based inversion.
# Runs `Sleipner_inversion_full_scale_weight_patch.py` under torchrun.
#
# >>> PATHS YOU MUST REPLACE BEFORE RUNNING <<<
#   conda activate <env>      : replace `opl2` with the name of YOUR conda env
#                               (the one created from environment.yml).
#   --ckpt_path               : path to the trained UNO3D checkpoint (.ckpt)
#                               produced by train_HNO_inversion_multi_gpu.py.
#                               Replace with the checkpoint on YOUR machine.
#   --segy_file               : path to the Sleipner SEGY cube to invert
#                               (e.g. one of 94p..., 99p..., 04p..., 06p... .sgy).
#                               Replace with the SEGY file on YOUR machine.
#   --back_model_npy          : path to the background impedance model (.npy).
#                               Replace with the background-model file on YOUR
#                               machine.
#   --out_dir                 : directory where partial / merged inversion
#                               results will be written. Replace with a
#                               writable directory on YOUR machine.
#
# Patch / optimization flags (--pd, --ph, --overlap_d, --overlap_h, --gain,
# --lr, --n_steps, --w_tv) and the GPU count (--nproc_per_node) can be tuned;
# the defaults below are the ones used in the paper.
# =============================================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate opl2

python -m torch.distributed.run \
  --nproc_per_node=2 \
  Sleipner_inversion_full_scale_weight_patch.py \
  --ckpt_path        /home/max0b/IP_inversion_NO/test_inversion1/test_inversion1-2025-12-07-15-41-25/checkpoints/epoch=22-step=20309.ckpt \
  --segy_file        /home/max0b/IP_inversion_NO/data/06p07ful.sgy \
  --back_model_npy   /home/max0b/IP_inversion_NO/4DProximal/data/Sleipner/Sleipner_impedance_model.npy \
  --out_dir          /home/max0b/IP_inversion_NO/Paper/out_sleipner_ddp_year2006 \
  --pd 150 --ph 234 --overlap_d 20 --overlap_h 20 \
  --gain 3 --lr 50 --n_steps 100 --w_tv 5

# Example for selecting GPUs explicitly:
# CUDA_VISIBLE_DEVICES=0,1,2,3 sh Sleipner_inversion_full_scale_weight_patch.sh
