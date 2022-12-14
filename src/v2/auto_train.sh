#! /bin/bash
python segment.py --n_epochs 100 --init_type 'normal' --criterion 'crossentropy' --save_root 'output/seg/8'
python segment.py --n_epochs 200 --init_type 'normal' --criterion 'crossentropy' --save_root 'output/seg/10'
python segment.py --n_epochs 100 --init_type 'kaiming' --criterion 'crossentropy' --save_root 'output/seg/11'
python segment.py --n_epochs 100 --init_type 'xavier' --criterion 'crossentropy' --save_root 'output/seg/12'
python segment.py --n_epochs 100 --init_type 'normal' --criterion 'focalloss' --save_root 'output/seg/13'
python segment.py --n_epochs 100 --init_type 'kaiming' --criterion 'focalloss' --save_root 'output/seg/14'
python segment.py --n_epochs 100 --init_type 'xavier' --criterion 'focalloss' --save_root 'output/seg/15'
python segment.py --n_epochs 100 --init_type 'normal' --criterion 'diceloss' --save_root 'output/seg/16'
python segment.py --n_epochs 100 --init_type 'kaiming' --criterion 'diceloss' --save_root 'output/seg/17'
python segment.py --n_epochs 100 --init_type 'xavier' --criterion 'diceloss' --save_root 'output/seg/18'
