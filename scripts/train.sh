#!/bin/bash

# Number of copies to run simultaneously
NUM_COPIES=2

eval "$(conda shell.bash hook)"
conda activate c_r_l

# Loop through the desired number of copies
for ((i=1; i<=$NUM_COPIES; i++)); do
    # Run the Python program with a different seed for each copy
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${i} python3 training.py --seed ${i} &
done

# Wait for all background processes to finish
wait

echo "All processes have finished."