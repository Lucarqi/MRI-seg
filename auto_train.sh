#! /bin/bash
python train.py --cuda --source_domain 'C0' --n_epochs 300 --save_root 'output/cyclegan/8' --decay_epoch 150
python train.py --cuda --source_domain 'T2' --n_epochs 200 --save_root 'output/cyclegan/9' --decay_epoch 100
python train.py --cuda --source_domain 'T2' --n_epochs 300 --save_root 'output/cyclegan/10' --decay_epoch 150