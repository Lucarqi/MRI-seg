python segment.py --batchSize 16 --n_epochs 100 --histogram_match true --init_type 'normal' --criterion 'diceloss' --save_root 'output/seg/'
python predict.py --histogram_match true --model_save 'output/seg/best_dice.pth' --results 'output/seg/results.txt'