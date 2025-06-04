#!/usr/bin/sh
for path in datasets/MATH/train/*; do
    echo "Running "${path##*/}
    python3 train.py --sector ${path##*/}
    echo "=================================================="
done
