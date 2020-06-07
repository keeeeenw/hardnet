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

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_test/ --model-variant=reshardnetdefaultsmall2 --epochs=1 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_adam_no_lr_change_lr01/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --optimizer=adam --change-lr=False --lr=0.1 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout03_kaiming_test/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --initialization=kaiming --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaulttiny_hardnet_liberty_train_with_aug_dropout03_kaiming/ --model-variant=reshardnetdefaulttiny --dropout=0.3 --epochs=10 --initialization=kaiming --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilev2_hardnet_liberty_train_with_aug_kaiming/ --model-variant=mobilenet_v2 --epochs=10 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnettiny_liberty_train_with_aug_test/ --model-variant=hardnettiny --epochs=10 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnetlarge_liberty_train_with_aug_test/ --model-variant=hardnetlarge --epochs=10 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=reshardnetdefaultsmallfc_liberty_train_with_aug_test/ --model-variant=reshardnetdefaultsmallfc --epochs=10 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=reshardnetdefaultsmall2fc_liberty_train_with_aug_droup03_test/ --model-variant=reshardnetdefaultsmall2fc --dropout=0.3 --epochs=10 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_reduced_liberty_train_with_aug_droup03_test/ --model-variant=mobilenet_v2_reduced --epochs=10 --n-triplets=5000 --enable-logging=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_tiny_liberty_train_with_aug_droup03_test/ --model-variant=mobilenet_v2_tiny --epochs=10 --n-triplets=5000 --enable-logging=False

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=densenet_reduced_liberty_train_with_aug_test/ --model-variant=densenet_reduced --epochs=10 --n-triplets=5000 --enable-logging=False