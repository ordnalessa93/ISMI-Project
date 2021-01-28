#!/usr/bin/env zsh

python trainer.py --network 'initial' --multi-gpu --epochs 200 \
    --aug 0.5 --batch-size 60 --filename 'models/first_model_all_augments_no_decay' \
    --optimizer 'adam' --aug-hflip --aug-vflip --aug-rotate 90 \
    --aug-brightness --drop-rate 0.2 --decay 0
