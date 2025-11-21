echo "Starting first job..."

CUDA_VISIBLE_DEVICES=7 python3 run.py crl --env ant_u_maze --total-env-steps 100000000 --goal-proposal-prob 0.5 --goal-proposer-name replay_buffer --use-adaptive-mixing --adaptive-mixing-momentum 0.999 > run1.out 2>&1

CUDA_VISIBLE_DEVICES=7 python3 run.py crl --env ant_u_maze --total-env-steps 100000000 --goal-proposer-name replay_buffer --goal-proposal-prob 0.25 > run2.out 2>&1


echo "All jobs completed."