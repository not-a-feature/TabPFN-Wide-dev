#!/bin/bash

# Default arguments
DATA_DIR="../benchmark_data/snp_data"
OUTPUT_FILE="../analysis_results/snp_benchmark_results.csv"
CHECKPOINTS="v2"
DEVICE="cuda:0"

# Run Benchmark
echo "Running SNP Benchmark..."
python analysis/snp_benchmark.py "$DATA_DIR" "$OUTPUT_FILE" --checkpoint_path "$CHECKPOINTS" --device "$DEVICE"

# Plot Results
echo "Plotting Results..."
python analysis/plot_snp_results.py "$OUTPUT_FILE" --output_dir "../analysis_results/plots"

echo "Done."
