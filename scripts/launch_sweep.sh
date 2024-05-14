#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <sweep_name> <num_runs>"
    exit 1
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp" && kill $(jobs -p)' EXIT

NAME="$1"
NUM="$2"

wandb sweep scripts/sweep.yml --name $NAME 2>&1 | tee >(sed -En 's/.*Run sweep agent with: (.*)/\1/p' > "$tmp")
grep ... "$tmp" || exit 1


for i in $(seq 1 $NUM); do
    cmd="$ENV $(cat "$tmp")"
    echo "$cmd"
    eval "$cmd" &
done

wait

