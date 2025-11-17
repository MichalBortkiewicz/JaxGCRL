CUDA_VISIBLE_DEVICES=4 python3 run.py crl --env ant --total-env-steps 80000000 --goal-proposer-name replay_buffer --use-adaptive-mixing --adaptive-mixing-warmup-steps 1000000 > run3.out 2>&1

CUDA_VISIBLE_DEVICES=4 python3 run.py crl --env ant_u_maze --total-env-steps 80000000 --goal-proposer-name replay_buffer --use-adaptive-mixing --adaptive-mixing-warmup-steps 1000000 > run4.out 2>&1