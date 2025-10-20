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
TOP_P=${TOP_P:-""}
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
    echo -e "${BLUE}Eval-Only Mode: Sanitize + Evaluate${NC}"
    echo "----------------------------------------------------------------------"
    echo "Input file: $EVAL_ONLY"
    echo ""

    # Check if already sanitized
    if [[ "$EVAL_ONLY" == *"-sanitized.jsonl" ]]; then
        echo "File already sanitized, proceeding to evaluation..."
        SANITIZED_INPUT="$EVAL_ONLY"
    else
        # Sanitize first
        echo "Step 1: Sanitizing..."
        docker run --rm \
            -v $(pwd):/app \
            ganler/evalplus:latest \
            evalplus.sanitize --samples "/app/$EVAL_ONLY"

        SANITIZED_INPUT="${EVAL_ONLY%.jsonl}-sanitized.jsonl"

        if [ ! -f "$SANITIZED_INPUT" ]; then
            echo -e "${RED}Error: Sanitized file not created${NC}"
            exit 1
        fi
        echo ""
        echo "Sanitized output: $SANITIZED_INPUT"
        echo ""
    fi

    # Evaluate
    echo "Step 2: Evaluating..."
    docker run --rm \
        -v $(pwd):/app \
        ganler/evalplus:latest \
        evalplus.evaluate --dataset humaneval \
        --samples "/app/$SANITIZED_INPUT"

    # Post-process results
    EVAL_RESULTS="${SANITIZED_INPUT%.jsonl}_eval_results.json"
    if [ -f "$EVAL_RESULTS" ]; then
        echo ""
        echo "Step 3: Processing results..."

        # Prettify the eval results JSON
        python3 -m json.tool "$EVAL_RESULTS" > "${EVAL_RESULTS%.json}_pretty.json"
        echo -e "${GREEN}✅ Prettified results saved${NC}"

        # Extract metrics summary
        python3 extract_metrics.py "$EVAL_RESULTS"
        echo -e "${GREEN}✅ Metrics summary extracted${NC}"
    fi

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

TOP_P_FLAG=""
if [ -n "$TOP_P" ]; then
    TOP_P_FLAG="--top-p $TOP_P"
fi

python3 -u eval_humaneval.py \
    --model $MODEL \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --num-samples $NUM_SAMPLES \
    $OUTPUT_DIR_FLAG \
    $TOP_P_FLAG \
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

# Build run name for subfolder
# Convert temperature to integer format (0.8 -> 80, 0.2 -> 20)
TEMP_INT=$(python3 -c "print(int(float('$TEMPERATURE') * 100))")
TEMP_STR=$(printf "t%02d" $TEMP_INT)

# Add top_p to run name if specified
if [ -n "$TOP_P" ]; then
    TOPP_INT=$(python3 -c "print(int(float('$TOP_P') * 100))")
    TOPP_STR=$(printf "_p%02d" $TOPP_INT)
else
    TOPP_STR=""
fi

RUN_NAME="samples_${TEMP_STR}${TOPP_STR}_n${NUM_SAMPLES}"
RUN_DIR="$RESULT_DIR/$RUN_NAME"
SAMPLES_FILE="$RUN_DIR/${RUN_NAME}.jsonl"

echo ""
echo -e "${GREEN}✅ Generation completed!${NC}"
echo "Output: $SAMPLES_FILE"
echo ""

# Step 2: Sanitize the generated code
echo -e "${BLUE}Step 2: Sanitizing generated code...${NC}"
echo "----------------------------------------------------------------------"

docker run --rm \
    -v $(pwd):/app \
    ganler/evalplus:latest \
    evalplus.sanitize --samples "/app/$SAMPLES_FILE"

SANITIZED_FILE="${SAMPLES_FILE%.jsonl}-sanitized.jsonl"

if [ ! -f "$SANITIZED_FILE" ]; then
    echo -e "${RED}Error: Sanitized file not created${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Sanitization completed!${NC}"
echo "Sanitized output: $SANITIZED_FILE"
echo ""

# Step 3: Evaluate with evalplus
echo -e "${BLUE}Step 3: Evaluating with evalplus...${NC}"
echo "----------------------------------------------------------------------"

docker run --rm \
    -v $(pwd):/app \
    ganler/evalplus:latest \
    evalplus.evaluate --dataset humaneval \
    --samples "/app/$SANITIZED_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Evaluation failed${NC}"
    exit 1
fi

# Step 4: Post-process results
echo ""
echo -e "${BLUE}Step 4: Post-processing results...${NC}"
echo "----------------------------------------------------------------------"

EVAL_RESULTS="${SANITIZED_FILE%.jsonl}_eval_results.json"
if [ -f "$EVAL_RESULTS" ]; then
    # Prettify the eval results JSON
    python3 -m json.tool "$EVAL_RESULTS" > "${EVAL_RESULTS%.json}_pretty.json"
    echo -e "${GREEN}✅ Prettified results saved${NC}"

    # Extract metrics summary
    python3 extract_metrics.py "$EVAL_RESULTS"
    echo -e "${GREEN}✅ Metrics summary extracted${NC}"

    # Show summary
    echo ""
    cat "${EVAL_RESULTS%.json}_summary.json"
else
    echo -e "${YELLOW}Warning: Eval results file not found: $EVAL_RESULTS${NC}"
fi

echo ""
echo -e "${GREEN}✅ End-to-end evaluation completed successfully!${NC}"
echo "======================================================================"
echo ""
echo "All files saved in: $RUN_DIR"
