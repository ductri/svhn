#!/bin/sh

nvidia-docker run -v `pwd`:/source/ \
    -v /root/code/all_dataset:/all_dataset
    --name trind_svhn trind/svhn \
    /bin/bash -c "python train.py --ALL_DATASET_PATH=/all_dataset"
