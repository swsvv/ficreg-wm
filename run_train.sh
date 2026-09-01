#!/bin/bash

usage() {
    echo "Usage: $0 <command template>"
    echo ""
    echo "Runs the command once per seed, replacing SEEDNUM with each seed value."
    echo ""
    echo "Example:"
    echo "  $0 python train.py data=pusht output_model_name=lewm-baseline-SEEDNUM seed=SEEDNUM"
    exit 1
}

[ $# -eq 0 ] && usage

# Active seeds for this run
SEEDS=(3072 42 101 3927 374024391 1702442591 751238365 1593226693 217519846 183184942 456748450)

CMD_TEMPLATE="$*"

for SEED in "${SEEDS[@]}"; do
    CMD="${CMD_TEMPLATE//SEEDNUM/$SEED}"

    echo "🔥 Running: $CMD"
    eval "$CMD"
    echo ""
done
