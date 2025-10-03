#!/usr/bin/env sh
shots=2
seed=42
now=$(date +"%Y%m%d_%H%M%S")

base_dir=YOUR_PATH
output_dir=${base_dir}/output_dir/ucf101_${shots}shot
mkdir -p ${output_dir}

# Step 1: 生成 few-shot list
python ${base_dir}/make_fewshot_list.py \
    --src_list caption/ucf101/ucf101_train_split_3_videos.txt \
    --dst_list ${base_dir}/output_dir/ucf101/train_list_${shots}shot.txt \
    --shots ${shots} \
    --seed ${seed}

# Step 2: 训练（finetune）
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 47709 \
    ${base_dir}/main.py \
    --model clip_vit_base_patch16_multimodal_adapter12x384 \
    --save_dir ${output_dir} \
    --dataset ucf101 \
    --num_frames 8 \
    --few_shot ${shots} \
    --sampling_rate 4 \
    --resize_type random_short_side_scale_jitter \
    --scale_range 1.0 1.15 \
    --num_spatial_views 1 \
    --num_temporal_views 3 \
    --label_csv 'caption/ucf101/ucf_101_labels.csv' \
    --mlm_label 'lables/k400_mlm_lables.txt' \
    --batch_size 16 \
    --epochs 15 \
    --warmup_epochs 2 \
    --eval_freq 1 \
    --print_freq 100 \
    --num_workers 14 \
    --lr 3.125e-4 \
    --mlm 1 \
    2>&1 | tee ${output_dir}/$now.log