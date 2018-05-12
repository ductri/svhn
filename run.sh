#!/bin/bash

for KERNEL_SIZE in {2..10..2}
do
    python train.py --CONV1_KERNEL_SIZE=$KERNEL_SIZE
done
