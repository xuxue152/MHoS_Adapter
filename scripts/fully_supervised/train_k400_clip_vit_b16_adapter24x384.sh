#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
base_dir=YOUR_PATH
dataset=k400
output_dir=${base_dir}/output_dir/${dataset}
mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 47708 \
    ${base_dir}/main.py \
    --model clip_vit_base_patch16_multimodal_adapter24x384 \
    --save_dir ${output_dir} \
    --auto_resume \
    --auto_remove \
    --dataset k400 \
    --num_frames 8 \
    --sampling_rate 16 \
    --resize_type random_short_side_scale_jitter \
    --scale_range 1.0 1.15 \
    --num_spatial_views 4 \
    --num_temporal_views 3 \
    --label_csv "lables/kinetics_400_labels.csv" \
    --mlm_label "lables/k400_mlm_lables.txt" \
    --mirror \
    --blr 5e-3 \
    --batch_size 8 \
    --epochs 12 \
    --warmup_epochs 2 \
    --eval_freq 12 \
    2>&1 | tee ${output_dir}/${now}.log