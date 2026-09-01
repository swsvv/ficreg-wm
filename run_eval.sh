#!/bin/bash

# Evaluate all checkpoints across 3 seeds
# Usage: bash run_eval.sh <model_prefix_with_SEEDNUM> <eval_config> <data_choice>
# Example: bash run_eval.sh lewm-baseline-SEEDNUM pusht.yaml pusht

if [ "$#" -ne 3 ]; then
    echo "Usage: bash $0 <model_prefix> <eval_config> <data_choice>"
    echo "Use SEEDNUM as placeholder for the seed value."
    echo "  data_choice: the Hydra data= value used during training (e.g. pusht, dmc)"
    echo "Example: bash $0 lewm-baseline-SEEDNUM pusht.yaml pusht"
    exit 1
fi

MODEL_PREFIX_TEMPLATE=$1
CONFIG=$2
DATA_CHOICE=$3

SEEDS=(3072 42 101 3927 374024391 1702442591 751238365 1593226693 217519846 183184942 456748450)
WM_DIR="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"

for SEED in "${SEEDS[@]}"; do
    MODEL_PREFIX="${MODEL_PREFIX_TEMPLATE//SEEDNUM/$SEED}"
    CKPT_DIR="${WM_DIR}/${DATA_CHOICE}_${MODEL_PREFIX}"

    echo "======================================"
    echo " Evaluating all checkpoints for:"
    echo " Seed          : $SEED"
    echo " Model prefix  : $MODEL_PREFIX"
    echo " Config        : $CONFIG"
    echo " CKPT dir      : $CKPT_DIR"
    echo " eval.py       : $(pwd)/eval.py"
    echo "======================================"
    echo ""

    if [ ! -d "$CKPT_DIR" ]; then
        echo "WARNING: Checkpoint directory not found: $CKPT_DIR — skipping seed $SEED"
        echo ""
        continue
    fi

    CKPT_FILES=$(find "$CKPT_DIR" -maxdepth 1 -name "*_object.ckpt" | sort -V)

    if [ -z "$CKPT_FILES" ]; then
        echo "No checkpoints matching '*_object.ckpt' found in $CKPT_DIR."
        echo ""
        continue
    fi

    for CKPT_FILE in $CKPT_FILES; do
        BASENAME=$(basename "$CKPT_FILE")
        POLICY_NAME="${BASENAME%_object.ckpt}"

        echo "[seed=$SEED $POLICY_NAME] Running eval..."

        OUTPUT=$(python eval.py \
            --config-name=${CONFIG} \
            policy=${DATA_CHOICE}_${MODEL_PREFIX}/${POLICY_NAME} 2>&1)

        SUCCESS_RATE=$(echo "$OUTPUT" | grep -oP "'success_rate':\s*\K[0-9.]+")

        if [ -n "$SUCCESS_RATE" ]; then
            echo "[seed=$SEED $POLICY_NAME] success_rate: ${SUCCESS_RATE}%"
        else
            echo "[seed=$SEED $POLICY_NAME] Could not parse success_rate. Raw output:"
            echo "$OUTPUT" | tail -5
        fi

        echo ""
    done
done

echo "======================================"
echo " Evaluation Pipeline Complete (all seeds)."
echo "======================================"
