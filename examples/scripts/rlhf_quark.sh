output_dir=$1
run_name=$2
reward_model_name_or_path=$3
policy_model_name_or_path=$4
kl_coef=${5:-0.05}

config_file="./examples/accelerate_configs/rlhf_ppo_fsdp_llama_4gpu.yaml"

accelerate launch --config_file "${config_file}" examples/rlhf_quark.py \
  --run_name "${run_name}" \
  --step_per_device_batch_size 2 \
  --rollout_per_device_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --output_dir "${output_dir}" \
  --reward_model_name_or_path "${reward_model_name_or_path}" \
  --policy_model_name_or_path "${policy_model_name_or_path}" \
  --rollout_batch_size 512 \
  --step_batch_size 256 \
  --learning_rate 3e-6 \
  --warmup_steps 5 \
  --kl_coef "${kl_coef}" \
  --total_epochs 10 \
  --flash_attn True \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --save_steps 20
