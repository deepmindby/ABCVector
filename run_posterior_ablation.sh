#!/bin/bash
# ==============================================================================
# ABC Vector Posterior Ablation Experiments
# ==============================================================================
#
# Runs four posterior_mode ablation experiments on GSM8K:
#   1) q_y_qca  — full privileged info (default)
#   2) q_y_qc   — CoT only, no answer leak
#   3) q_y_q    — question features only
#   4) none     — prior-only training
#
# All experiments save training diagnostics + diagnostic eval comparison.
#
# Usage:
#   chmod +x run_posterior_ablation.sh
#   ./run_posterior_ablation.sh
#
# Customize the variables below before running.
# ==============================================================================

# ===== Configuration =====
MODEL_PATH="/home/haichao/TA/ABCVector/models/Qwen2.5-Math-7B"
MODEL_NAME="qwen"
DATA_PATH="/home/haichao/TA/ABCVector/data"
DATASET="gsm8k"
LAYER_IDX=0
LR="5e-5"
KL_BETA=0.05
NUM_EPOCHS=5
NUM_SUPPORT=3000
NUM_TEST=500
OUTPUT_DIR="./outputs"

# Common arguments
COMMON_ARGS="--method abc \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --dataset $DATASET \
    --layer_idx $LAYER_IDX \
    --abc_learning_rate $LR \
    --kl_beta $KL_BETA \
    --num_epochs $NUM_EPOCHS \
    --num_support_samples $NUM_SUPPORT \
    --num_test_samples $NUM_TEST \
    --output_dir $OUTPUT_DIR \
    --save_diagnostics \
    --run_prior_eval \
    --diagnostic_split both \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_length 1024"

echo "======================================================================"
echo "ABC Vector Posterior Ablation"
echo "======================================================================"
echo "Model:      $MODEL_PATH"
echo "Dataset:    $DATASET"
echo "Layer:      $LAYER_IDX"
echo "LR:         $LR"
echo "KL beta:    $KL_BETA"
echo "Epochs:     $NUM_EPOCHS"
echo "Support:    $NUM_SUPPORT"
echo "Test:       $NUM_TEST"
echo "======================================================================"

MODES=("q_y_qca" "q_y_qc" "q_y_q" "none")

for MODE in "${MODES[@]}"; do
    echo ""
    echo "======================================================================"
    echo ">>> Running posterior_mode = $MODE"
    echo "======================================================================"
    
    EXTRA_ARGS=""
    if [ "$MODE" != "none" ]; then
        EXTRA_ARGS="--run_posterior_eval"
    fi
    
    python main.py $COMMON_ARGS \
        --posterior_mode $MODE \
        $EXTRA_ARGS \
        2>&1 | tee "outputs/${DATASET}/ablation_${MODE}.log"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: posterior_mode=$MODE failed with exit code $EXIT_CODE"
    else
        echo "DONE: posterior_mode=$MODE completed successfully"
    fi
done

echo ""
echo "======================================================================"
echo "All ablation experiments complete."
echo "Results saved to: outputs/${DATASET}/diagnostics/"
echo "======================================================================"
echo ""
echo "Diagnostics files:"
ls -la outputs/${DATASET}/diagnostics/ 2>/dev/null || echo "(no diagnostics yet)"