#!/bin/bash
# This script is used for multi-gpu training, inference and 
# benchmarking the PSPNet with SegSort on PASCAL VOC 2012.
#
# Usage:
#   # From SegSort/ directory.
#   bash bashscripts/voc12/train_segsort_mgpu.sh


# Set up parameters for training. 

#tw: large epochs, 256 batch size

BATCH_SIZE=256
TRAIN_INPUT_SIZE=60,60
WEIGHT_DECAY=1e-4
NUM_EPOCHS1=300
NUM_CLASSES=1000
NUM_GPU=2
LEARNING_RATE=0.1
NUM_LOADING_WORKERS=4

# Set up parameters for inference.
# INFERENCE_INPUT_SIZE=480,480
# INFERENCE_STRIDES=320,320
# INFERENCE_SPLIT=val

# Set up SegSort hyper-parameters.
EMBEDDING_DIM=32
KMEANS_ITERATIONS=10
K_IN_NEAREST_NEIGHBORS=21

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/imgnet/unsup_embed/unsup_embed_lr1e-1_it10k

# Set up the procedure pipeline.
IS_TRAIN_INET_1=1

# Set up the data directory.
DATAROOT_EMBED=snapshots/voc12/unsup_segsort/unsup_segsort_lr2e-3_it10k/embeds/32

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH


# Train ImageNet for first stage.
# Run unsup_segsort.sh before this to get checkpoint model
  #qq: how does pytorch do model saving?
#qq: what is the eq. use global status in pytorch?
#qq: this should be defined right
# dang: all the spaces after the \ caused args not to parse correctly
if [ ${IS_TRAIN_INET_1} -eq 1 ]; then
  python3 -u pyscripts/train/train_embed_imgnet.py\
        --data_dir ${DATAROOT_EMBED}\
        --batch_size ${BATCH_SIZE}\
        --snapshot_dir ${SNAPSHOT_DIR}/stage1\
        --save_pred_every $(($NUM_EPOCHS1/4))\
        --update_tb_every 50\
        --use_global_status\
        --input_size ${TRAIN_INPUT_SIZE}\
        --learning_rate ${LEARNING_RATE}\
        --weight_decay ${WEIGHT_DECAY}\
        --num_classes ${NUM_CLASSES}\
        --num_epochs $(($NUM_EPOCHS1+1))\
        --num_gpu ${NUM_GPU}\
        --embedding_dim ${EMBEDDING_DIM}\
        --random_mirror\
        --random_scale\
        --random_crop\
        --is_training
fi
