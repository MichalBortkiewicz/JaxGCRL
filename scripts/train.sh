#!/bin/bash

# Number of copies to run simultaneously
NUM_COPIES=3

# Loop through the desired number of copies
for ((i=1; i<=$NUM_COPIES; i++)); do
    # Run the Python program with a different seed for each copy
    CUDA_VISIBLE_DEVICES=3 python training.py  &
done

# Wait for all background processes to finish
wait

echo "All processes have finished."