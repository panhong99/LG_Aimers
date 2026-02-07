#!/bin/bash
set -e
cd /home/hail/pan/LG_AI
sleep 5
echo "Starting training with master_port 29501..."
conda run -n LG_AI deepspeed --master_port 29501 --num_gpus=2 train_kd.py --method kd --epochs 1
