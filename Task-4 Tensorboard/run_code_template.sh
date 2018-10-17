#!/bin/bash
export TF_LOG_DIR="tf/log/"

mkdir -p ~/tf/log
python lr.py
tensorboard --logdir $TF_LOG_DIR