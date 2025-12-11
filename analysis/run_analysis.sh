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
if [ ! -f "${OUTPUT_DIR}/hdlss_benchmark_results.csv" ]; then
    echo "----------------------------------------"
    echo "Running HDLSS Benchmark..."
    echo "----------------------------------------"
    python analysis/hdlss_benchmark.py \
        "benchmark_data/hdlss_new_data" \
        "${OUTPUT_DIR}/hdlss_benchmark_results.csv" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --config_path "$CONFIG_FILE"
else
    echo "HDLSS Benchmark results exist. Skipping."
fi

if [ "$RUN_HDLSS_ONLY" = "true" ]; then
    echo "HDLSS only mode enabled. Skipping other benchmarks."
    echo "----------------------------------------"
    echo "Plotting Results (HDLSS Only)..."
    echo "----------------------------------------"
    python analysis/plot_results.py --input_dir "$OUTPUT_DIR"
    exit 0
fi

# Multi-omics Feature Reduction
if [ ! -f "${OUTPUT_DIR}/multiomics_feature_reduction_results.csv" ]; then
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
else
    echo "Multi-omics Feature Reduction results exist. Skipping."
fi

# OpenML Benchmark
if [ ! -f "${OUTPUT_DIR}/openml_benchmark_results.csv" ]; then
    echo "----------------------------------------"
    echo "Running OpenML Benchmark..."
    echo "----------------------------------------"
    python analysis/openml_benchmark.py \
        "${OUTPUT_DIR}/openml_benchmark_results.csv" \
        --suite_id 457 \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --config_path "$CONFIG_FILE"
else
    echo "OpenML Benchmark results exist. Skipping."
fi

# OpenML Widening
if [ ! -d "${OUTPUT_DIR}/openml_widening" ]; then
    echo "----------------------------------------"
    echo "Running OpenML Widening..."
    echo "----------------------------------------"
    python analysis/openml_widening.py \
        "${OUTPUT_DIR}/openml_widening" \
        --dataset_ids "1494 40536" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --config_path "$CONFIG_FILE"
else
    echo "OpenML Widening results directory exists. Skipping."
fi

# Multi-omics Attention Extraction
if [ ! -f "${OUTPUT_DIR}/multiomics_attention.pt" ]; then
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
else
    echo "Multi-omics Attention Extraction results exist. Skipping."
fi

# Widening Attention Extraction
if [ ! -f "${OUTPUT_DIR}/widening_attention.pkl" ]; then
    echo "----------------------------------------"
    echo "Running Widening Attention Extraction..."
    echo "----------------------------------------"
    python analysis/extract_widening_attention.py \
        "${OUTPUT_DIR}/widening_attention.pkl" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --config_path "$CONFIG_FILE" \
        --openml_id 1494
else
    echo "Widening Attention Extraction results exist. Skipping."
fi

# Plotting Results
echo "----------------------------------------"
echo "Plotting Results..."
echo "----------------------------------------"
python analysis/plot_results.py --input_dir "$OUTPUT_DIR"

echo "All analysis completed."
