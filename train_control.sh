#!/bin/bash

MODEL_DIR="runwayml/stable-diffusion-v1-5"  # 修改为你的模型路径
OUTPUT_DIR="outputs"

accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --root=RGBT-CC \
    --resolution=384 \
    --learning_rate=2e-5 \
    --validation_image "RGBT-CC/test/depth_t/1236_T.jpg" "RGBT-CC/test/depth_t/1517_T.jpg" "RGBT-CC/test/depth_t/1742_T.jpg" "RGBT-CC/test/depth_t/3405_T.jpg" "RGBT-CC/test/depth_t/2809_T.jpg" "RGBT-CC/test/depth_t/1363_T.jpg" "RGBT-CC/val/depth_t/1290_T.jpg" "RGBT-CC/val/depth_t/1394_T.jpg" "RGBT-CC/val/depth_t/2080_T.jpg" "RGBT-CC/val/depth_t/3713_T.jpg"\
    --validation_prompt "" "" "" "" "" "" "" "" "" ""\
    --train_batch_size=8 \
    --num_train_epochs=500 \
    --max_train_steps=150000 \
    --validation_steps=100 \
    --gradient_accumulation_steps=16 \
    --checkpoints_total_limit=5 \
    --num_validation_images=1 \
    --controlnet_model_name_or_path="lllyasviel/control_v11f1p_sd15_depth"
