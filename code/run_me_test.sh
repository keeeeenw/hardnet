#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/reshardnet_logs"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"


( # Download and prepare data
    cd "$DATASETS"
    if [ ! -d "wxbs-descriptors-benchmark/data/W1BS" ]; then
        git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
        chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        ./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
        mv W1BS wxbs-descriptors-benchmark/data/
        rm -f W1BS*.tar.gz
    fi
)


cd "$RUNPATH"

# Reshardnet default
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefault_hardnet_liberty_train_with_aug_test/ --model-variant=reshardnetdefault --epochs=10 --n-triplets=5000

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_test/ --model-variant=reshardnet34 --epochs=10 --n-triplets=5000 --dropout=0.3 --enable-logging=False

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_test/ --model-variant=reshardnetdefaultsmall --epochs=10 --n-triplets=5000 --dropout=0.3 --enable-logging=False