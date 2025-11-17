echo "Starting first job..."

CUDA_VISIBLE_DEVICES=6 python3 run.py crl --env ant_u_maze --total-env-steps 120000000 --goal-proposer-name metric --goal-proposal-prob 0.5 > run1.out 2>&1

CUDA_VISIBLE_DEVICES=6 python3 run.py crl --env ant --total-env-steps 80000000 --goal-proposer-name metric --goal-proposal-prob 0.5 > run1.out 2>&1

echo "All jobs completed."