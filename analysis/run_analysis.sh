#!/bin/bash

set -e

CHECKPOINT_PATH="$1"
OUTPUT_DIR="$2"
RUN_HDLSS_ONLY="${3:-false}"

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <checkpoint_path> <output_dir> [run_hdlss_only]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
CONFIG_FILE="${CHECKPOINT_DIR}/config.json"

echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Using config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

# HDLSS Benchmark
echo "----------------------------------------"
echo "Running HDLSS Benchmark..."
echo "----------------------------------------"
python analysis/hdlss_benchmark.py \
    "benchmark_data/hdlss_new_data" \
    "${OUTPUT_DIR}/hdlss_benchmark_results.csv" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE"

if [ "$RUN_HDLSS_ONLY" = "true" ]; then
    echo "HDLSS only mode enabled. Skipping other benchmarks."
    exit 0
fi

# Multi-omics Feature Reduction
echo "----------------------------------------"
echo "Running Multi-omics Feature Reduction..."
echo "----------------------------------------"
python analysis/multiomics_feature_reduction.py \
    "benchmark_data/multiomics_benchmark_data" \
    "${OUTPUT_DIR}/multiomics_feature_reduction_results.csv" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE" \
    --dataset "BRCA" \
    --omics "mRNA cnv"

# OpenML Benchmark
echo "----------------------------------------"
echo "Running OpenML Benchmark..."
echo "----------------------------------------"
python analysis/openml_benchmark.py \
    "${OUTPUT_DIR}/openml_benchmark_results.csv" \
    --suite_id 457 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE"

# OpenML Widening
echo "----------------------------------------"
echo "Running OpenML Widening..."
echo "----------------------------------------"
python analysis/openml_widening.py \
    "${OUTPUT_DIR}/openml_widening" \
    --dataset_ids "1494 40536" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE"

# Multi-omics Attention Extraction
echo "----------------------------------------"
echo "Running Multi-omics Attention Extraction..."
echo "----------------------------------------"
python analysis/extract_multi_omics_attention.py \
    "benchmark_data/multiomics_benchmark_data" \
    "${OUTPUT_DIR}/multiomics_attention.pt" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE" \
    --dataset "BRCA" \
    --omic "mrna"

# Widening Attention Extraction
echo "----------------------------------------"
echo "Running Widening Attention Extraction..."
echo "----------------------------------------"
python analysis/extract_widening_attention.py \
    "${OUTPUT_DIR}/widening_attention.pkl" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_FILE" \
    --openml_id 1494

echo "All analysis completed."
