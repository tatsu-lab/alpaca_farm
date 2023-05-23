python examples/best_of_n.py \
  --task "run_decode" \
  --decoder_name_or_path "/juice5/scr5/nlp/crfm/human-feedback/models/selfinstruct/sft_v6_llama_7b_regen_v7_3ep" \
  --output_path "./tmp.json"

python examples/best_of_n.py \
  --task "run_best_of_n" \
  --decoder_name_or_path "/juice5/scr5/nlp/crfm/human-feedback/models/selfinstruct/sft_v6_llama_7b_regen_v7_3ep" \
  --scorer_name_or_path "/juice5/scr5/nlp/crfm/human-feedback/models/selfinstruct/reward_model_scaling_v3_for_sft_v6_llama_7b_regen_v7_3ep/reward_model_v3_train_size_19382"

python examples/best_of_n.py \
  --task "run_best_of_n" \
  --decoder_name_or_path "/juice5/scr5/nlp/crfm/human-feedback/models/selfinstruct/sft_v6_llama_7b_regen_v7_3ep" \
  --scorer_name_or_path "/juice5/scr5/nlp/crfm/human-feedback/models/selfinstruct/reward_model_scaling_v3_for_sft_v6_llama_7b_regen_v7_3ep/reward_model_v3_train_size_19382" \
  --num_return_sequences 16 \
  --per_device_batch_size 4 \
  --split "eval" \
  --mixed_precision "bf16" \
  --tf32 True \
  --flash_attn True \
  --output_path "./bon.json" \
  --max_instances 10
