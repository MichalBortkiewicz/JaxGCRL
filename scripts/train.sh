#!/bin/bash

# Number of copies to run simultaneously
NUM_COPIES=4

eval "$(conda shell.bash hook)"
conda activate c_r_l

# Loop through the desired number of copies
for bs in 512 1024 2048; do
  for ((i=1; i<=$NUM_COPIES; i++)); do
      # Run the Python program with a different seed for each copy
      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=3 python training.py --num_evals 50 --seed 1 --num_timesteps 500000 --batch_size ${bs} --exp_name "r_bs_${bs}"
  done
done

# Wait for all background processes to finish
#wait

echo "All processes have finished."