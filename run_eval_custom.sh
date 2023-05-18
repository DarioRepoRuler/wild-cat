#!/bin/bash

python annotator.py
GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 configs/OWOD_eval.sh
python API_manager.py --engine google
python show_web_results.py