config_file=$1
reward_model_name_or_path=$2
policy_model_name_or_path=$3
output_dir=$4

accelerate launch --config_file "${config_file}" examples/rlhf_ppo.py \
  --run_name "rlhf_ppo" \
  --step_per_device_batch_size 2 \
  --rollout_per_device_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --output_dir "${output_dir}" \
  --reward_model_name_or_path "${reward_model_name_or_path}" \
  --policy_model_name_or_path "${policy_model_name_or_path}" \
  --init_value_with_reward True \
  --rollout_batch_size 512 \
  --step_batch_size 256 \
  --learning_rate 1e-5 \
  --warmup_steps 5 \
  --kl_coef 0.002 \
  --total_epochs 10 \
  --flash_attn True \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --save_steps 20
