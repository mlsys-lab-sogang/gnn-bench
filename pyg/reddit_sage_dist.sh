#!/usr/bin/env bash
python3 reddit_sage_dist.py --fanout 15 10 5 --batch_size 1024 --node_id $RANK
