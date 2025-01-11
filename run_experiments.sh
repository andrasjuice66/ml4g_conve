#!/bin/bash

# Base directory for datasets
DATA_DIR="data"

# Common parameters that might need adjustment
DEVICE="cuda"  # Change to "cpu" if no GPU available
NUM_WORKERS=4  # Adjust based on your CPU cores
LEARNING_RATE=0.001

# Function to run an experiment
run_experiment() {
    local dataset=$1
    local batch_size=$2
    local epochs=$3
    local emb_dim=$4
    local label_smoothing=$5

    echo "Running experiment on ${dataset}..."
    python -m main \
        --data_dir "${DATA_DIR}/${dataset}" \
        --batch_size ${batch_size} \
        --epochs ${epochs} \
        --embedding_dim ${emb_dim} \
        --learning_rate ${LEARNING_RATE} \
        --label_smoothing ${label_smoothing} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS}
}

# FB15k-237 (smaller dataset)
run_experiment "FB15k-237" \
    128 \    # batch_size
    100 \    # epochs
    200 \    # embedding_dim
    0.1      # label_smoothing

# WN18RR (smaller dataset, might need more epochs due to complexity)
run_experiment "WN18RR" \
    128 \    # batch_size
    200 \    # epochs
    200 \    # embedding_dim
    0.1      # label_smoothing

# YAGO3-10 (larger dataset)
run_experiment "YAGO3-10" \
    256 \    # batch_size
    100 \    # epochs
    300 \    # embedding_dim
    0.1      # label_smoothing 