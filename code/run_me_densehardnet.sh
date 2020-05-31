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

# SGD, 10 epoch
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=densenet121_hardnet_liberty_train/ --model-variant=densenet121 --epochs=10
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=densenet121_hardnet_liberty_train_with_aug/ --model-variant=densenet121 --epochs=10

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=densenet121_hardnet_notredame_train/ --training-set=notredame --model-variant=densenet121 --epochs=10
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=densenet121_hardnet_notredame_train_with_aug/ --training-set=notredame --model-variant=densenet121 --epochs=10

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=densenet121_hardnet_yosemite_train/ --training-set=yosemite  --model-variant=densenet121 --epochs=10
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=densenet121_hardnet_yosemite_train_with_aug/ --training-set=yosemite --model-variant=densenet121 --epochs=10

