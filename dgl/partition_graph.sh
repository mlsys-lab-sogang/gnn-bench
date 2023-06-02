#!/usr/bin/env bash
python3 partition_graph.py --num_parts $NUM_PARTS --num_trainers_per_machine $NUM_GPUS --balance_train --balance_edges --output_dir ./dataset/reddit_partition