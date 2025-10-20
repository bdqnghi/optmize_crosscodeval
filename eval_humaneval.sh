#!/bin/bash
#
# HumanEval Evaluation Script
# End-to-end generation and evaluation using evalplus
#

set -e

# Configuration
MODEL=${MODEL:-"3b"}
TEMPERATURE=${TEMPERATURE:-0.2}
MAX_TOKENS=${MAX_TOKENS:-512}
NUM_SAMPLES=${NUM_SAMPLES:-1}
OUTPUT_DIR=${OUTPUT_DIR:-""}
NO_CONTEXT=${NO_CONTEXT:-""}
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
if ! python3 -c "import datasets" 2>/dev/null; then
    echo -e "${YELLOW}Warning: datasets library not found${NC}"
    echo "Installing datasets..."
    pip install datasets
    echo ""
fi

# Handle eval-only mode
if [ -n "$EVAL_ONLY" ]; then
    echo -e "${BLUE}Evaluating existing results with evalplus...${NC}"
    echo "----------------------------------------------------------------------"
    echo "Input file: $EVAL_ONLY"
    echo ""

    docker run --rm \
        -v $(pwd):/app \
        ganler/evalplus:latest \
        evalplus.evaluate --dataset humaneval \
        --samples "/app/$EVAL_ONLY"

    echo ""
    echo -e "${GREEN}✅ Evaluation completed!${NC}"
    echo "======================================================================"
    exit 0
fi

# Step 1: Generate completions
echo -e "${BLUE}Step 1: Generating completions...${NC}"
echo "----------------------------------------------------------------------"

CONTEXT_FLAG=""
if [ "$NO_CONTEXT" = "true" ]; then
    CONTEXT_FLAG="--no-context"
fi

OUTPUT_DIR_FLAG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_FLAG="--output-dir $OUTPUT_DIR"
fi

python3 -u eval_humaneval.py \
    --model $MODEL \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --num-samples $NUM_SAMPLES \
    $OUTPUT_DIR_FLAG \
    $CONTEXT_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Generation failed${NC}"
    exit 1
fi

# Get the output file path
if [ -n "$OUTPUT_DIR" ]; then
    RESULT_DIR="$OUTPUT_DIR"
else
    # Auto-detect based on model
    python3 -c "
import json
config = json.load(open('config.json'))
model = '$MODEL'
if model in config['models']:
    print('results/humaneval/' + config['models'][model]['name'])
else:
    print('results/humaneval/' + model)
" > /tmp/result_dir.txt
    RESULT_DIR=$(cat /tmp/result_dir.txt)
    rm /tmp/result_dir.txt
fi

TEMP_STR=$(printf "t%02d" $((${TEMPERATURE%.*} * 10 + ${TEMPERATURE#*.})))
SAMPLES_FILE="$RESULT_DIR/samples_${TEMP_STR}_n${NUM_SAMPLES}.jsonl"

echo ""
echo -e "${GREEN}✅ Generation completed!${NC}"
echo "Output: $SAMPLES_FILE"
echo ""

# Step 2: Evaluate with evalplus
echo -e "${BLUE}Step 2: Evaluating with evalplus...${NC}"
echo "----------------------------------------------------------------------"

docker run --rm \
    -v $(pwd):/app \
    ganler/evalplus:latest \
    evalplus.evaluate --dataset humaneval \
    --samples "/app/$SAMPLES_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Evaluation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ End-to-end evaluation completed successfully!${NC}"
echo "======================================================================"
