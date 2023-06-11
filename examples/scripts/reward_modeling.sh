output_dir=$1
run_name=$2
model_name_or_path=$3
seed=${4:-42}
flash_attn=${5:-True}
dataset_name=${6:-"alpaca_noisy_multi_preference"}

torchrun --nproc_per_node=4 --master_port=64536 examples/reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --seed "${seed}" \
  --model_name_or_path "${model_name_or_path}" \
  --dataset_name "${dataset_name}" \
  --output_dir "${output_dir}" \
  --model_max_length 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --eval_steps 50 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 3e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm_flash_vs_nonflash" \
  --run_name "${run_name}" \
  --fsdp "full_shard auto_wrap offload" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
  --tf32 True \
  --flash_attn "${flash_attn}" \
  --ddp_timeout 1800
