#!/bin/bash
#
# HumanEval Evaluation Script
# Unified generation and evaluation
#

set -e

# Configuration
MODEL=${MODEL:-"3b"}
TEMPERATURE=${TEMPERATURE:-0.2}
MAX_TOKENS=${MAX_TOKENS:-512}
NUM_SAMPLES=${NUM_SAMPLES:-1}
OUTPUT_DIR=${OUTPUT_DIR:-""}
NO_CONTEXT=${NO_CONTEXT:-""}
GENERATE_ONLY=${GENERATE_ONLY:-""}
EVAL_ONLY=${EVAL_ONLY:-""}

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "HumanEval Evaluation Pipeline"
echo "======================================================================"
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Max Tokens: $MAX_TOKENS"
echo "Num Samples: $NUM_SAMPLES"
echo "======================================================================"
echo ""

# Check dependencies
if ! python3 -c "import evalplus" 2>/dev/null; then
    echo -e "${YELLOW}Warning: evalplus library not found${NC}"
    echo "Installing evalplus..."
    pip install evalplus
    echo ""
fi

if ! python3 -c "import datasets" 2>/dev/null; then
    echo -e "${YELLOW}Warning: datasets library not found${NC}"
    echo "Installing datasets..."
    pip install datasets
    echo ""
fi

# Run unified evaluation
echo -e "${BLUE}Running HumanEval evaluation...${NC}"
echo "----------------------------------------------------------------------"

CONTEXT_FLAG=""
if [ "$NO_CONTEXT" = "true" ]; then
    CONTEXT_FLAG="--no-context"
fi

OUTPUT_DIR_FLAG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_FLAG="--output-dir $OUTPUT_DIR"
fi

GENERATE_ONLY_FLAG=""
if [ "$GENERATE_ONLY" = "true" ]; then
    GENERATE_ONLY_FLAG="--generate-only"
fi

EVAL_ONLY_FLAG=""
if [ -n "$EVAL_ONLY" ]; then
    EVAL_ONLY_FLAG="--eval-only $EVAL_ONLY"
fi

python3 -u eval_humaneval.py \
    --model $MODEL \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --num-samples $NUM_SAMPLES \
    $OUTPUT_DIR_FLAG \
    $CONTEXT_FLAG \
    $GENERATE_ONLY_FLAG \
    $EVAL_ONLY_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Evaluation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
echo "======================================================================"
