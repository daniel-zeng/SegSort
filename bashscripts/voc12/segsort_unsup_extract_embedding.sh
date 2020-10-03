#!/bin/bash
# This script is used for multi-gpu training, inference and 
# benchmarking the PSPNet with SegSort on PASCAL VOC 2012.
#
# Usage:
#   # From SegSort/ directory.
#   bash bashscripts/voc12/train_segsort_mgpu.sh


# Set up parameters for training. 

#tw: large epochs, 256 batch size


BATCH_SIZE=16
TRAIN_INPUT_SIZE=480,480
NUM_CLASSES=1000
NUM_GPU=2

# Set up SegSort hyper-parameters.
EMBEDDING_DIM=32

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/voc12/unsup_segsort/unsup_segsort_lr2e-3_it10k

# Set up the procedure pipeline.
IS_EXTRACT_1=1



# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory.
DATAROOT=/home/dz/SegSort/dataset/
DATAROOT_IMGNET=/home/public/public_dataset/ILSVRC2014/Img/

# Train ImageNet for first stage.
# Run unsup_segsort.sh before this to get checkpoint model
if [ ${IS_EXTRACT_1} -eq 1 ]; then
  python3 pyscripts/inference/extract_embeddings.py\
    --save_dir ${SNAPSHOT_DIR}/embeddings\
    --snapshot_dir ${SNAPSHOT_DIR}/stage2\
    --restore_from ${SNAPSHOT_DIR}/stage1/model.ckpt-10000\
    --data_dir ${DATAROOT_IMGNET}\
    --batch_size ${BATCH_SIZE}\
    --use_global_status\
    --input_size ${TRAIN_INPUT_SIZE}\
    --num_classes ${NUM_CLASSES}\
    --num_gpu ${NUM_GPU}\
    --embedding_dim ${EMBEDDING_DIM}\
    --is_training
fi