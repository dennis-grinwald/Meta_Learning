#!/bin/sh

python run_maml.py --n_way=5 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1
python run_maml.py --n_way=5 --k_shot=1 --inner_update_lr=0.04 --num_inner_updates=1
python run_maml.py --n_way=5 --k_shot=1 --inner_update_lr=4 --num_inner_updates=1
