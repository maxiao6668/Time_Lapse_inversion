#!/bin/bash
# =============================================================================
# Multi-GPU training launcher for the 3D Hierarchical Neural Operator (HNO).
#
# >>> PATHS YOU MUST REPLACE BEFORE RUNNING <<<
#   --results_path     : directory where checkpoints / logs will be saved.
#                        Replace with a writable directory on YOUR machine.
#   --exp_name         : experiment name (used as a sub-folder under results_path
#                        and as the WandB run name). Change to whatever you want.
#   --train_data_path  : path to the training set (folder that contains the
#                        synthetic seismic / impedance pairs). Replace with the
#                        location where you stored the training data.
#   --valid_data_path  : path to the validation set (same format as the training
#                        set). Replace with your validation data folder.
#
# Other flags (--gpus, --batch_size, --lr, --epochs) can be tuned to match your
# hardware; the values below are the ones used in the paper on 4 x A100.
# =============================================================================

nohup python train_HNO_inversion_multi_gpu.py \
  --results_path     /home/max0b/IP_inversion_NO/test_inversion_2025_12_17_3D_small \
  --exp_name         test_inversion_2025_12_17_3D_small \
  --gpus             3 \
  --train_data_path  /data/max0b/synthoseis_example/2025_11_23_3D_data_imp/train/ \
  --valid_data_path  /data/max0b/synthoseis_example/2025_11_23_3D_data_imp/val/ \
  --batch_size       1 \
  --lr               1e-3 \
  --epochs           40 \
  > output_no.log 2>&1 &
