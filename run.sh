#!/bin/bash

#DATA_DIR=/home/kmyoon/data/LJSpeech-1.1
#WORK_DIR=/home/kmyoon/exp/tf-melgan
DATA_DIR=/home/feesh/corpus/LJSpeech-1.1
WORK_DIR=/home/feesh/projects/tf-melgan
OUT_DIR=${WORK_DIR}/outputs

train_tfrecord=${OUT_DIR}/ljspeech/train.tfrecord
valid_tfrecord=${OUT_DIR}/ljspeech/valid.tfrecord

if [ ! -f ${train_tfrecord} ] || [ ! -f ${valid_tfrecord} ]; then
    mkdir -p ${OUT_DIR}/ljspeech
    python3 prepare_ljspeech.py \
        --data_dir ${DATA_DIR} \
        --train_tfrecord ${train_tfrecord} \
        --valid_tfrecord ${valid_tfrecord} \
        --sampling_rate 22050 \
        --frame_length_in_sec 0.02 \
        --step_length_in_sec 0.011 \
        --num_fft_point 512 \
        --hertz_low 0 \
        --hertz_high 11025 \
        || exit 1;
fi

python3 train_melgan.py \
    --train_tfrecord ${train_tfrecord} \
    --valid_tfrecord ${valid_tfrecord} \
    --log_dir ${OUT_DIR}/logs \
    --chkpt_dir ${OUT_DIR}/chkpts \
    --sampling_rate 22050 \
    --frame_length_in_sec 0.02 \
    --step_length_in_sec 0.011 \
    --num_fft_point 512 \
    --hertz_low 0 \
    --hertz_high 11025 \
    --max_duration 10 \
    --discriminator_num_blocks 3 \
    --discriminator_leaky_relu_alpha 0.2 \
    --batch_size 4 \
    --optimizer adam \
    --init_learning_rate 0.0001 \
    --num_iter 100 \
    --use_hinge_loss true \
    || exit 1;