#! /bin/bash



cd PATH_TO_FOLDER/GNN_trk
OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 top_tagging.py --exp_name=300_500 --datadir='./data/300_500' --logdir='./logs/' --batch_size=16
