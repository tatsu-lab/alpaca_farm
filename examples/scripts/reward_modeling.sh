output_dir=$1
run_name=$2
model_name_or_path=$3
dataset_name=${4:-"alpaca_noisy_multi_preference"}

torchrun --nproc_per_node=8 --master_port=1234 examples/reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --model_name_or_path "${model_name_or_path}" \
  --dataset_name "${dataset_name}" \
  --output_dir "${output_dir}" \
  --model_max_length 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_steps 10 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 3e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm" \
  --run_name "${run_name}" \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
  --tf32 True \
  --flash_attn True \
  --ddp_timeout 1800
