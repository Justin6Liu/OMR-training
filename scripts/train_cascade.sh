#!/bin/bash
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omr

CONFIG=${1:-configs/cascade_omr.py}

python -m mmdet.apis.train $CONFIG
