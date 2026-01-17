#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:.

CHECKPOINT_PATH="$1"
OUTPUT_DIR="$2"
RUN_HDLSS_ONLY="${3:-false}"
RUN_HDLSS_ONLY="${3:-false}"
MAX_FEATURES="${4:-50000}"

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <checkpoint_path> <output_dir> [run_hdlss_only] [max_features]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ "$CHECKPOINT_PATH" == "default_n1g1" ] || [ "$CHECKPOINT_PATH" == "stock" ] || [ "$CHECKPOINT_PATH" == "v2" ] || [[ "$CHECKPOINT_PATH" == wide-v2* ]]; then
    echo "Using default/valid model: $CHECKPOINT_PATH"
    CONFIG_FILE=""
else
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
    CONFIG_FILE="${CHECKPOINT_DIR}/config.json"
    echo "Using checkpoint: $CHECKPOINT_PATH"
    echo "Using config: $CONFIG_FILE"
fi

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
        --config_path "$CONFIG_FILE" \
        --max_features "$MAX_FEATURES"
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
    for dataset in BRCA COAD GBM LGG OV; do
        echo "Processing dataset: $dataset"
        python analysis/multiomics_feature_reduction.py \
            "benchmark_data/multiomics_benchmark_data" \
            "${OUTPUT_DIR}/multiomics_feature_reduction_results.csv" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --config_path "$CONFIG_FILE" \
            --dataset "$dataset" \
            --omics "mrna"
    done
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
        --dataset_ids 1494 40536 \
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
    for dataset in BRCA COAD GBM LGG OV; do
        echo "Processing dataset: $dataset"
        python analysis/extract_multi_omics_attention.py \
            "benchmark_data/multiomics_benchmark_data" \
            "${OUTPUT_DIR}/multiomics_attention.pt" \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --config_path "$CONFIG_FILE" \
            --dataset "$dataset" \
            --omic "mrna"
    done
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

# SNP Benchmark
if [ ! -f "${OUTPUT_DIR}/snp_benchmark_results.csv" ]; then
    echo "----------------------------------------"
    echo "Running SNP Benchmark..."
    echo "----------------------------------------"
    python analysis/snp_benchmark.py \
        "benchmark_data/snp_data" \
        "${OUTPUT_DIR}/snp_benchmark_results.csv" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --device "cuda:0"
else
    echo "SNP Benchmark results exist. Skipping."
fi

# Plotting Results
echo "----------------------------------------"
echo "Plotting Results..."
echo "----------------------------------------"
python analysis/plot_results.py --input_dir "$OUTPUT_DIR"

echo "All analysis completed."
