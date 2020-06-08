#!/bin/bash
RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/logs"
DATAMODELS="$RUNPATH/data/models"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"
mkdir -p "$DATAMODELS"

cd "$RUNPATH"
python -utt ./code/HardNetMultipleDatasets.py --training-set=all  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_6Brown_30M_reshardnetdefaultsmall2final" --dataroot "$DATASETS" --lr=10.0 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=30000000 --imageSize 32 --log-dir "$DATALOGS" --experiment-name=reshardnetdefaultsmall2final_brown6_aug_lr10/ --model-variant=reshardnetdefaultsmall2

python -utt ./code/HardNetMultipleDatasets.py --training-set=all  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_6Brown_30M_resdefaulttiny" --dataroot "$DATASETS" --lr=10.0 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=30000000 --imageSize 32 --log-dir "$DATALOGS" --experiment-name=resdefaulttiny_brown6_aug_lr10/ --model-variant=reshardnetdefaulttiny

python -utt ./code/HardNetMultipleDatasets.py --training-set=all  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_6Brown_30M_reshardnet34" --dataroot "$DATASETS" --lr=10.0 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=30000000 --imageSize 32 --log-dir "$DATALOGS" --experiment-name=reshardnet34_brown6_aug_lr10/ --model-variant=reshardnet34

python -utt ./code/HardNetMultipleDatasets.py --training-set=all  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_6Brown_30M_hardnetlarge" --dataroot "$DATASETS" --lr=10.0 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=30000000 --imageSize 32 --log-dir "$DATALOGS" --experiment-name=hardnetlarge_brown6_aug_lr10/ --model-variant=hardnetlarge

python -utt ./code/HardNetMultipleDatasets.py --training-set=all  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_6Brown_30M_reshardnet" --dataroot "$DATASETS" --lr=10.0 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=30000000 --imageSize 32 --log-dir "$DATALOGS" --experiment-name=reshardnet_brown6_aug_lr10/ --model-variant=reshardnet

python -utt ./code/HardNetMultipleDatasets.py --training-set=all  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_6Brown_30M_ortho06_lr10" --dataroot "$DATASETS" --lr=10.0 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=30000000 --imageSize 32 --log-dir "$DATALOGS" --experiment-name=brown6_aug_lr10/