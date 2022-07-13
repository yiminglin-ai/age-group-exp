#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python train.py experiment=fairface-dex model.backbone.model_name=resnet18,resnet34 -m
python train.py experiment=fairface-coral model.backbone.model_name=resnet18,resnet34 -m
