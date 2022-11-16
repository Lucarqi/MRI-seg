python segment.py --batchSize 8 --n_epochs 100 --init_type 'normal' --model 'unet' --criterion 'crossentropy' --save_root 'output/seg/43'
python predict.py --model 'unet'   --model_save 'output/seg/43/best_dice.pth' --results 'output/seg/43/results.txt'
python segment.py --batchSize 8 --n_epochs 100 --init_type 'kaiming' --model 'munet' --criterion 'crossentropy' --save_root 'output/seg/44'
python predict.py --model 'unet'   --model_save 'output/seg/44/best_dice.pth' --results 'output/seg/44/results.txt'