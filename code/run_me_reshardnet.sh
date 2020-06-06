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

# Reshardnetsmall2
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall2_hardnet_liberty_train_with_aug_dropout/ --model-variant=reshardnetdefaultsmall2 --dropout=0.3

# Dropout enabled
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_05/ --model-variant=reshardnet34 --epochs=10 --dropout=0.5

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_adam/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --optimizer=adam 

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_adam_r0001/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --optimizer=adam --lr=0.001

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_r01/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --lr=0.1

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_dropout_03_r001/ --model-variant=reshardnet34 --epochs=10 --dropout=0.3 --lr=0.01

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

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_001/ --model-variant=reshardnet34 --epochs=10 --lr=0.01

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_0001/ --model-variant=reshardnet34 --epochs=10 --lr=0.001

# # Different batch sizes
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_batch_128/ --optimizer=adam --model-variant=reshardnet34 --epochs=3 --batch-size=128
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_batch_512/ --optimizer=adam --model-variant=reshardnet34 --epochs=3 --batch-size=512

# # ResNet34, SGD, 10 epoch, larger learning rate
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res34_hardnet_liberty_train_with_aug_sgd_01/ --model-variant=reshardnet34 --epochs=3 --lr=30

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

# ResNet50, SGD, 10 epoch
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=res50_hardnet_liberty_train_with_aug/ --model-variant=reshardnet50 --epochs=10

# Parameter search for resdefaultsmall - best model 06/03 with ~0.0068 FPR95 for notredame
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout03/ --model-variant=reshardnetdefaultsmall --epochs=10 --dropout=0.3

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug/ --model-variant=reshardnetdefaultsmall --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_adam/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --optimizer=adam --change-lr=False

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_adam_no_lr_change_lr01/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --optimizer=adam --change-lr=False --lr=0.1
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_adam_no_lr_change_lr001/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --optimizer=adam --change-lr=False --lr=0.01
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_adam/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --optimizer=adam

# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_droupout_lr_01/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --lr=0.1
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_lr_001/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --lr=0.01
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout_droupout_eps_20/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=20

# # mobilenet v2
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilev2_hardnet_liberty_train_with_aug_kaiming/ --model-variant=mobilenet_v2 --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilev2_hardnet_liberty_train_with_aug_kaiming_adam/ --model-variant=mobilenet_v2 --epochs=10 --optimizer=adam

# # resdefaulttiny
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaulttiny_hardnet_liberty_train_with_aug/ --model-variant=reshardnetdefaulttiny --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaulttiny_hardnet_liberty_train_with_aug_dropout03/ --model-variant=reshardnetdefaulttiny --dropout=0.3 --epochs=10

# hardnet tiny, 16 suffix has the correct filter size
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnettiny16_liberty_train_with_aug/ --model-variant=hardnettiny --epochs=10 --dropout=0.0
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnettiny16_liberty_train_with_aug_drouput03/ --model-variant=hardnettiny --epochs=10 --dropout=0.3
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnettiny16_liberty_train_with_aug_lr01/ --model-variant=hardnettiny --epochs=10 --dropout=0.0 --lr=0.1
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnettiny16_liberty_train_with_aug_lr001/ --model-variant=hardnettiny --epochs=10 --dropout=0.0 --lr=0.01
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnettiny16_liberty_train_with_aug_adam --model-variant=hardnettiny --epochs=10 --dropout=0.0 --optimizer=adam

# hardnet large
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnetlarge_liberty_train_with_aug/ --model-variant=hardnetlarge --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnetlarge_liberty_train_with_aug_drouput03/ --model-variant=hardnetlarge --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnetlarge_liberty_train_with_aug_lr01/ --model-variant=hardnetlarge --epochs=10 --lr=0.1
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnetlarge_liberty_train_with_aug_lr001/ --model-variant=hardnetlarge --epochs=10 --lr=0.01
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=hardnetlarge_liberty_train_with_aug_adam --model-variant=hardnetlarge --epochs=10 --optimizer=adam

# mobilenet reduced
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_reduced_liberty_train_with_aug/ --model-variant=mobilenet_v2_reduced --epochs=10
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_reduced_liberty_train_with_aug_adam/ --model-variant=mobilenet_v2_reduced --epochs=10 --optimizer=adam
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_reduced_liberty_train_with_aug_lr01/ --model-variant=mobilenet_v2_reduced --epochs=10 --lr=0.1
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_reduced_liberty_train_with_aug_lr001/ --model-variant=mobilenet_v2_reduced --epochs=10 --lr=0.01

# mobilenet tiny
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_tiny_liberty_train_with_aug/ --model-variant=mobilenet_v2_tiny --epochs=10
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_tiny_liberty_train_with_aug_adam/ --model-variant=mobilenet_v2_tiny --epochs=10 --optimizer=adam
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_tiny_liberty_train_with_aug_lr01/ --model-variant=mobilenet_v2_tiny --epochs=10 --lr=0.1
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=mobilenet_v2_tiny_liberty_train_with_aug_lr001/ --model-variant=mobilenet_v2_tiny --epochs=10 --lr=0.01

# reshardnetsmall fc layer
# python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=reshardnetdefaultsmallfc_liberty_train_with_aug/ --model-variant=reshardnetdefaultsmallfc --epochs=10
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=reshardnetdefaultsmallfc_liberty_train_with_aug_dropout03/ --model-variant=reshardnetdefaultsmallfc --dropout=0.3 --epochs=10
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=reshardnetdefaultsmall2fc_liberty_train_with_aug/ --model-variant=reshardnetdefaultsmall2fc --epochs=10
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=reshardnetdefaultsmall2fc_liberty_train_with_aug_dropout03/ --model-variant=reshardnetdefaultsmall2fc --dropout=0.3 --epochs=10

# resdefaultsmall with kaiming initialization
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout03_kaiming/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --initialization=kaiming
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout03_kaiming_lr01/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --initialization=kaiming --lr=0.1
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout03_kaiming_adam/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --initialization=kaiming --optimizer=adam
python ./code/HardNet.py --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --fliprot=True --experiment-name=resdefaultsmall_hardnet_liberty_train_with_aug_dropout03_kaiming_adam_lr01/ --model-variant=reshardnetdefaultsmall --dropout=0.3 --epochs=10 --initialization=kaiming --optimizer=adam --lr=0.1