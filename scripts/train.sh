#!/bin/bash

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

eval "$(conda shell.bash hook)"
conda activate contrastive_rl

env=humanoid

for seed in 1; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python training.py \
    --project_name test --group_name first_run --exp_name test --num_evals 10 \
    --seed ${seed} --num_timesteps 1000000 --batch_size 256 --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env_name ${env} \
    --episode_length 1000 --unroll_length 62  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --multiplier_num_sgd_steps 1 --log_wandb
  done

echo "All runs have finished."