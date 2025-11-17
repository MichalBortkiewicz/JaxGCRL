echo "Starting first job..."

CUDA_VISIBLE_DEVICES=6 python3 run.py crl --env ant --total-env-steps 80000000 --goal-proposer-name replay_buffer --goal-proposal-prob 0.5 --use-adaptive-mixing > run1.out 2>&1

echo "All jobs completed."