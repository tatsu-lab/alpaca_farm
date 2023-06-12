import transformers

from alpaca_farm import utils


def test_stable_resize_token_embeddings():
    model_name_or_paths = (
        "gpt2",  # Tied weights.
        "/juice5/scr5/nlp/llama_model/llama_hf_latest/llama-teeny",  # Untied weights.
    )
    for model_name_or_path in model_name_or_paths:
        model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
        utils.stable_resize_token_embeddings(
            model, target_size=model.get_input_embeddings().weight.size(0) + 10, jitter_new_embeddings=True
        )
