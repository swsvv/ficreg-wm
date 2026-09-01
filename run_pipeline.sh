#!/bin/bash

# Run training then evaluation in sequence.
# Usage: bash run_pipeline.sh <eval_config> <train_command_template>
#
# Example:
#   bash run_pipeline.sh pusht.yaml \
#     python train.py data=pusht output_model_name=lewm-baseline-SEEDNUM seed=SEEDNUM

if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 <eval_config> <train_command_template...>"
    echo ""
    echo "Example:"
    echo "  bash $0 pusht.yaml python train.py data=pusht output_model_name=lewm-baseline-SEEDNUM seed=SEEDNUM"
    exit 1
fi

EVAL_CONFIG=$1
shift
TRAIN_CMD="$*"

# Extract the model name template (with SEEDNUM still in it)
MODEL_NAME_TEMPLATE=$(echo "$TRAIN_CMD" | grep -oP 'output_model_name=\K\S+')
DATA_CHOICE=$(echo "$TRAIN_CMD" | grep -oP 'data=\K\S+')

if [ -z "$MODEL_NAME_TEMPLATE" ]; then
    echo "ERROR: Could not parse 'output_model_name=' from the train command."
    exit 1
fi

if [ -z "$DATA_CHOICE" ]; then
    echo "ERROR: Could not parse 'data=' from the train command."
    exit 1
fi

echo "=== Pipeline Start ==="
echo " Train command : $TRAIN_CMD"
echo " Model template: $MODEL_NAME_TEMPLATE"
echo " Data choice   : $DATA_CHOICE"
echo " Eval config   : $EVAL_CONFIG"
echo "======================="
echo ""

bash run_train.sh $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed. Skipping evaluation."
    exit 1
fi

echo ""
echo "=== Training complete. Starting evaluation... ==="
echo ""

bash run_eval.sh "$MODEL_NAME_TEMPLATE" "$EVAL_CONFIG" "$DATA_CHOICE"
