#! /bin/bash
python segment.py --batchSize 8 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/43'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/43/best_dice.pth' --results 'output/seg/stage-1/43/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/44'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/44/best_dice.pth' --results 'output/seg/stage-1/44/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/45'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/45/best_dice.pth' --results 'output/seg/stage-1/45/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/46'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/46/best_dice.pth' --results 'output/seg/stage-1/46/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/47'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/47/best_dice.pth' --results 'output/seg/stage-1/47/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/48'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/48/best_dice.pth' --results 'output/seg/stage-1/48/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/49'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/49/best_dice.pth' --results 'output/seg/stage-1/49/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/50'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/50/best_dice.pth' --results 'output/seg/stage-1/50/results.txt' --histogram_match true
python segment.py --batchSize 8 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/51'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/15/best_dice.pth' --results 'output/seg/stage-1/51/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/52'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/52/best_dice.pth' --results 'output/seg/stage-1/52/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/53'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/53/best_dice.pth' --results 'output/seg/stage-1/53/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/54'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/54/best_dice.pth' --results 'output/seg/stage-1/54/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/55'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/55/best_dice.pth' --results 'output/seg/stage-1/55/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/56'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/56/best_dice.pth' --results 'output/seg/stage-1/56/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/57'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/57/best_dice.pth' --results 'output/seg/stage-1/57/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/58'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/58/best_dice.pth' --results 'output/seg/stage-1/58/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/59'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/59/best_dice.pth' --results 'output/seg/stage-1/59/results.txt' --histogram_match true
python segment.py --batchSize 4 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/60'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/60/best_dice.pth' --results 'output/seg/stage-1/60/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/61'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/61/best_dice.pth' --results 'output/seg/stage-1/61/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/27'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/62/best_dice.pth' --results 'output/seg/stage-1/62/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'crossentropy' --save_root 'output/seg/stage-1/63'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/63/best_dice.pth' --results 'output/seg/stage-1/63/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/64'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/64/best_dice.pth' --results 'output/seg/stage-1/64/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/65'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/65/best_dice.pth' --results 'output/seg/stage-1/65/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'focalloss' --save_root 'output/seg/stage-1/66'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/66/best_dice.pth' --results 'output/seg/stage-1/66/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'normal' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/67'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/67/best_dice.pth' --results 'output/seg/stage-1/67/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'kaiming' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/68'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/68/best_dice.pth' --results 'output/seg/stage-1/68/results.txt' --histogram_match true
python segment.py --batchSize 16 --n_epochs 100 --init_type 'xavier' --model 'unet' --histogram_match true --criterion 'diceloss' --save_root 'output/seg/stage-1/69'
python predict.py --model 'unet' --histogram_match true  --model_save 'output/seg/stage-1/69/best_dice.pth' --results 'output/seg/stage-1/69/results.txt' --histogram_match true

