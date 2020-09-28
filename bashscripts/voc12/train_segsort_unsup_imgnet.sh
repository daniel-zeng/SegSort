#!/bin/bash
# This script is used for multi-gpu training, inference and 
# benchmarking the PSPNet with SegSort on PASCAL VOC 2012.
#
# Usage:
#   # From SegSort/ directory.
#   bash bashscripts/voc12/train_segsort_mgpu.sh


# Set up parameters for training. 
BATCH_SIZE=64
TRAIN_INPUT_SIZE=240,240
WEIGHT_DECAY=5e-4
ITER_SIZE=1
NUM_STEPS1=10000
NUM_CLASSES=1000
NUM_GPU=2
LEARNING_RATE=1e-1

# Set up parameters for inference.
# INFERENCE_INPUT_SIZE=480,480
# INFERENCE_STRIDES=320,320
# INFERENCE_SPLIT=val

# Set up SegSort hyper-parameters.
CONCENTRATION=10
NUM_BANKS=2
EMBEDDING_DIM=32
NUM_CLUSTERS=5
KMEANS_ITERATIONS=10
K_IN_NEAREST_NEIGHBORS=21

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/voc12/unsup_segsort/unsup_segsort_lr2e-3_it10k

# Set up the procedure pipeline.
IS_TRAIN_INET_1=1

# IS_TRAIN_1=0
# IS_PROTOTYPE_1=0
# IS_INFERENCE_1=0
# IS_INFERENCE_MSC_1=0
# IS_BENCHMARK_1=0



# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory.
DATAROOT=/home/dz/SegSort/dataset/
DATAROOT_IMGNET=/home/public/public_dataset/ILSVRC2014/Img/

# Train ImageNet for first stage.
# Run unsup_segsort.sh before this to get checkpoint model
if [ ${IS_TRAIN_INET_1} -eq 1 ]; then
  python3 pyscripts/train/train_segsort_imagenet_class.py\
    --snapshot_dir ${SNAPSHOT_DIR}/stage2\
    --restore_from ${SNAPSHOT_DIR}/stage1/model.ckpt-10000\
    --data_dir ${DATAROOT_IMGNET}\
    --batch_size ${BATCH_SIZE}\
    --save_pred_every 10000\
    --update_tb_every 50\
    --use_global_status\
    --input_size ${TRAIN_INPUT_SIZE}\
    --learning_rate ${LEARNING_RATE}\
    --weight_decay ${WEIGHT_DECAY}\
    --iter_size ${ITER_SIZE}\
    --num_classes ${NUM_CLASSES}\
    --num_steps $(($NUM_STEPS1+1))\
    --num_gpu ${NUM_GPU}\
    --embedding_dim ${EMBEDDING_DIM}\
    --random_mirror\
    --random_scale\
    --random_crop\
    --is_training
fi