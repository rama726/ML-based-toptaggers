#! /bin/bash

cd PATH_TO_FOLDER/CNN_trk
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 model.py --exp_name=300_500 --datadir='./data/300_500' --logdir='./logs/' --num_block 3 3 3 --hidden 16 32 64 --num_workers=4 --lr=0.001 --batch_size=32
