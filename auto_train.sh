#! /bin/bash
python segment.py --batchSize 16 --n_epochs 100 --init_type 'normal' --model 'aunet'   --criterion 'diceloss' --save_root 'output/seg/47'
python predict.py --model 'aunet' --model_save 'output/seg/47/best_dice.pth' --results 'output/seg/47/results.txt'   