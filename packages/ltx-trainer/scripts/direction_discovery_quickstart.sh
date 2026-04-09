#!/bin/bash
# Quick-start script for direction discovery

set -e

echo "🚀 Direction Discovery Quick Start"
echo "=================================="
echo ""

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <video_path> <checkpoint_path> <text_encoder_path> <prompt> [method] [num_directions]"
    echo ""
    echo "Example:"
    echo "  $0 input.mp4 ltx2.safetensors /path/to/gemma \"cinematic scene\" random 10"
    echo ""
    exit 1
fi

VIDEO_PATH=$1
CHECKPOINT_PATH=$2
TEXT_ENCODER_PATH=$3
PROMPT=$4
METHOD=${5:-random}
NUM_DIRECTIONS=${6:-10}
OUTPUT_DIR="outputs/direction_discovery_$(date +%Y%m%d_%H%M%S)"

echo "Video: $VIDEO_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Text encoder: $TEXT_ENCODER_PATH"
echo "Prompt: $PROMPT"
echo "Method: $METHOD"
echo "Directions: $NUM_DIRECTIONS"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Video not found: $VIDEO_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$TEXT_ENCODER_PATH" ]; then
    echo "❌ Error: Text encoder directory not found: $TEXT_ENCODER_PATH"
    exit 1
fi

echo "⏳ Running direction discovery..."
python -m ltx_trainer.direction_discovery run-direction-discovery \
    "$VIDEO_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --text-encoder-path "$TEXT_ENCODER_PATH" \
    --prompt "$PROMPT" \
    --method "$METHOD" \
    --num-directions "$NUM_DIRECTIONS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "⏳ Analyzing results..."
python -m ltx_trainer.direction_discovery analyze-results \
    --metrics-path "$OUTPUT_DIR/metrics.json" \
    --top-k 5

echo ""
echo "✅ Done! Results saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  - Stability plot: $OUTPUT_DIR/stability_plot.png"
echo "  - Direction grids: $OUTPUT_DIR/direction_*_grid.png"
echo "  - Metric curves: $OUTPUT_DIR/direction_*_metrics.png"
echo "  - Raw metrics: $OUTPUT_DIR/metrics.json"