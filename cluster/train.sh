#!/bin/bash
PYTHON="/work/scratch/azad/anaconda3/envs/pytorch_cuda11/bin/python"
WORK_DIR="/home/staff/azad/deeplearning/afshin"

cd ${WORK_DIR}
chmod +x train.sh
$PYTHON ${WORK_DIR}/src/train.py "$@"