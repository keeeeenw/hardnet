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

# ( # Run the code
#     cd "$RUNPATH"
#     # python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=liberty_train/ --n-triplets $@ | tee -a "$DATALOGS/log_HardNet_Lib.log"
#     # python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=liberty_train_with_aug/  $@ | tee -a "$DATALOGS/log_HardNetPlus_Lib.log"
# )

cd "$RUNPATH"
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_liberty_train/ --model-variant=reshardnet --start-epoch=8 --resume=/home/ken/workspace/image-matching-benchmark-baselines/third_party/hardnet/data/models/res_hardnet_liberty_train/_liberty_min/checkpoint_7.pth
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res_hardnet_liberty_train_with_aug/ --model-variant=reshardnet
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=liberty_train_with_aug/  $@ | tee -a "$DATALOGS/log_HardNetPlus_Lib.log"

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_notredame_train/ --training-set=notredame --model-variant=reshardnet
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res_hardnet_notredame_train_with_aug/ --training-set=notredame --model-variant=reshardnet
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=notredame_train/ --training-set=notredame $@ | tee -a "$DATALOGS/notredame_log_HardNet_Lib.log"
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=notredame_train_with_aug/ --training-set=notredame $@ | tee -a "$DATALOGS/notredame_log_HardNetPlus_Lib.log"

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_yosemite_train/ --training-set=yosemite --model-variant=reshardnet
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res_hardnet_yosemite_train_with_aug/ --training-set=yosemite --model-variant=reshardnet
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=yosemite_train/ --training-set=yosemite $@ | tee -a "$DATALOGS/yosemite_log_HardNet_Lib.log"
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=yosemite_train_with_aug/ --training-set=yosemite $@ | tee -a "$DATALOGS/yosemite_log_HardNetPlus_Lib.log"

