#!/bin/bash

export PYTHONPATH="$(pwd)"

python3 main.py \
  --model_name="conv" \
  --reset_output_dir \
  --data_path="." \
  --output_dir="outputs" \
  --n_classes=10 \
  --train_steps=10000 \
  --batch_size=32 \
  --log_every=1000 \
  "$@"

