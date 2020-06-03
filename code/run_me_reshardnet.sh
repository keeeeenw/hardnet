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

# SCG, 10 epoch
# #python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_liberty_train/ --model-variant=reshardnet --start-epoch=8 --resume=/home/ken/workspace/image-matching-benchmark-baselines/third_party/hardnet/data/models/res_hardnet_liberty_train/_liberty_min/checkpoint_7.pth
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_liberty_train/ --model-variant=reshardnet 
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res_hardnet_liberty_train_with_aug/ --model-variant=reshardnet
# # python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=liberty_train_with_aug/  $@ | tee -a "$DATALOGS/log_HardNetPlus_Lib.log"

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_notredame_train/ --training-set=notredame --model-variant=reshardnet
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res_hardnet_notredame_train_with_aug/ --training-set=notredame --model-variant=reshardnet
# # python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=notredame_train/ --training-set=notredame $@ | tee -a "$DATALOGS/notredame_log_HardNet_Lib.log"
# # python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=notredame_train_with_aug/ --training-set=notredame $@ | tee -a "$DATALOGS/notredame_log_HardNetPlus_Lib.log"

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_yosemite_train/ --training-set=yosemite --model-variant=reshardnet
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res_hardnet_yosemite_train_with_aug/ --training-set=yosemite --model-variant=reshardnet
# # python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=yosemite_train/ --training-set=yosemite $@ | tee -a "$DATALOGS/yosemite_log_HardNet_Lib.log"
# # python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=yosemite_train_with_aug/ --training-set=yosemite $@ | tee -a "$DATALOGS/yosemite_log_HardNetPlus_Lib.log"

# Adam, 10 epoch
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_liberty_train_adam/ --model-variant=reshardnet --optimizer=adam --epochs=10

# # Reshardnet default with dropout
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefault_hardnet_liberty_train_with_aug_dropout/ --model-variant=reshardnetdefault --epochs=10 --dropout=0.3

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefault_hardnet_liberty_train_with_aug_dropout_adam/ --model-variant=reshardnetdefault --epochs=10 --optimizer=adam --dropout=0.3

# Reshardnet small
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout03/ --model-variant=reshardnetdefaultsmall --epochs=10 --dropout=0.3

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug/ --model-variant=reshardnetdefaultsmall --epochs=10

# ResNet50, SGD, 10 epoch
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res50_hardnet_liberty_train_with_aug/ --model-variant=reshardnet50 --epochs=10

# Dropout enabled
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_05/ --model-variant=reshardnet34 --epochs=10 --dropout=0.5

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_adam/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --optimizer=adam 

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_adam_r0001/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --optimizer=adam --lr=0.001

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_r01/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --lr=0.1

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_r001/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --lr=0.01

# # ResNet34, Adam, 10 epoch
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_adam/ --optimizer=adam --model-variant=reshardnet34 --epochs=10

# ResNet34, Adam, 10 epoch, no weight decay, smaller learning rate
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_adam_01/ --optimizer=adam --model-variant=reshardnet34 --epochs=10 --lr=0.1

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_adam_001/ --optimizer=adam --model-variant=reshardnet34 --epochs=10 --lr=0.01

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_adam_0001/ --optimizer=adam --model-variant=reshardnet34 --epochs=10 --lr=0.001

# ResNet34, SGD, 10 epoch, smaller learning rate
# loss NaN
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_01/ --optimizer=adam --model-variant=reshardnet34 --epochs=3 --lr=0.1

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_001/ --optimizer=adam --model-variant=reshardnet34 --epochs=3 --lr=0.01
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_001/ --model-variant=reshardnet34 #--epochs=7 --lr=0.01 --start-epoch=4 --resume=/home/ken/workspace/image-matching-benchmark-baselines/third_party/hardnet/data/models/res34_hardnet_liberty_train_with_aug_sgd_001/_liberty_min/checkpoint_2.pth

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_01/ --model-variant=reshardnet34 --epochs=10 --lr=0.1

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_001/ --model-variant=reshardnet34 --epochs=10 --lr=0.01

python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_0001/ --model-variant=reshardnet34 --epochs=10 --lr=0.001


# Different batch sizes
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_batch_128/ --optimizer=adam --model-variant=reshardnet34 --epochs=3 --batch-size=128
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_batch_512/ --optimizer=adam --model-variant=reshardnet34 --epochs=3 --batch-size=512

# ResNet34, SGD, 10 epoch, larger learning rate
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_01/ --model-variant=reshardnet34 --epochs=3 --lr=30

# ResNet34, SGD, 10 epoch
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res34_hardnet_liberty_train/ --model-variant=reshardnet34 --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug/ --model-variant=reshardnet34 --epochs=10

# # ResNet101, SGD, 10 epoch
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res101_hardnet_liberty_train_with_aug/ --model-variant=reshardnet101 --epochs=10

# # ResNet34, SGD, 10 epoch --training-set=notredame
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res34_hardnet_notredame_train/ --training-set=notredame --model-variant=reshardnet34 --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_notredame_train_with_aug/ --training-set=notredame --model-variant=reshardnet34 --epochs=10

# # ResNet34, SGD, 10 epoch --training-set=yosemite
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res34_hardnet_yosemite_train/ --training-set=yosemite --model-variant=reshardnet34 --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_yosemite_train_with_aug/ --training-set=yosemite --model-variant=reshardnet34 --epochs=10

# Adam, 10 epoch, resnet zero_init_residual True
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_liberty_train_adam/ --model-variant=reshardnet --optimizer=adam --epochs=5

# Pretrained, Adam, 3 epoch
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=False --experiment-name=res_hardnet_liberty_train_adam/ --model-variant=reshardnet --optimizer=adam --epochs=3 --pre-trained=True

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_test/ --model-variant=reshardnet34 --epochs=1 --n-triplets=5000 --enable-logging=False
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res_hardnet_liberty_train_with_aug_test/ --model-variant=reshardnet50 --epochs=1 --n-triplets=5000 --enable-logging=False
