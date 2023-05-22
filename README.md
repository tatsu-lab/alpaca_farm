<p align="center" width="100%">
<a href="https://crfm.stanford.edu/alpaca/" target="_blank"><img src="assets/AlpacaFarm_big.png" alt="AlpacaFarm" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# AlpacaFarm: A Simulation Framework for <br/>Methods that Learn from Human Feedback

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Research and development on learning from human feedback is difficult because methods
like [RLHF](https://arxiv.org/abs/2203.02155) are costly to run and complex to analyze.
AlpacaFarm is a simulator that enables research and development on learning from feedback at a fraction of the usual
cost,
promoting accessible research on instruction following and alignment.

This repo contains code for

- [simulating preference feedback from language models such as GPT-4](#simulating-pairwise-preference)
- [automated evaluation for instruction-following models](#running-automatic-evaluation)
- [validated reference implementations of baseline methods such as PPO and best-of-n](#running-reference-methods)

The data needed to run our code is hosted on HuggingFace: https://huggingface.co/datasets/tatsu-lab/alpaca_farm.

**Usage and License Notices**: Alpaca is intended and licensed for research use only. The dataset is CC BY NC 4.0 (
allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
The weight diff is also CC BY NC 4.0 (allowing only non-commercial use).

## The AlpacaFarm

<br>
<p style="text-align:center;">
  <img style="max-width:70%; height:auto;" src="./assets/fig1.jpg" alt="Workflow">
</p>

Instruction-following models are typically developed in 3 steps

1. Supervised fine-tuning with demonstrations
2. Learning from human feedback; usually pairwise preferences
3. Human evaluation with interaction

The goal of AlpacaFarm is to provide three key components that tackles steps 2 and 3:
Low-cost simulation of pairwise feedback from API models (e.g. GPT-4, ChatGPT), automated evaluations for methods
development, and reference implementations of
learning algorithms for comparison and modification.

## Installation

For basic installation, run

```bash
pip install git+https://github.com/tatsu-lab/alpaca_farm.git
```

To enable FlashAttention and other optimizations, install
the [`flash-attn`](https://github.com/HazyResearch/flash-attention) and [`apex`](https://github.com/NVIDIA/apex)
packages.

## Simulating Pairwise Preference

**Notebook example:** [![Using](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/lossyless/blob/main/notebooks/Hub.ipynb) 

<details>
  <summary><b>Installing auto annotators with minimal dependencies</b></summary>
    To install only the auto annotators with minimal additional packages use the following
    
    ```
    MINIMAL_DEPENDENCIES=1
    pip install alpacafarm[annotations]
    ```
</details>

To get started either go through the colab notebook above or run one of the two following commands depending on the format of your data.

<details>
<summary><b>Annotating list of pairs of outputs</b></summary>
<pre><code class="python">
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_farm.utils import jload

# load some data
outputs_pairs = jload("examples/data/outputs_pairs.json")[:6]
print(outputs_pairs[-1:])
# [{'instruction': 'If you could help me write an email to my friends inviting them to dinner on Friday, it would be greatly appreciated.',
#   'input': '',
#   'output_1': "Dear Friends, \r\n\r\nI hope this message finds you well. I'm excited to invite you to dinner on Friday. We'll meet at 7:00 PM at [location]. I look forward to seeing you there. \r\n\r\nBest,\r\n[Name]",
#   'output_2': "Hey everyone! \n\nI'm hosting a dinner party this Friday night and I'd love for all of you to come over. We'll have a delicious spread of food and some great conversations. \n\nLet me know if you can make it - I'd love to see you all there!\n\nCheers,\n[Your Name]"}]

annotator = PairwiseAutoAnnotator()
annotated = annotator.annotate_pairs(outputs_pairs)

print(annotated[-1:])
# [{'instruction': 'If you could help me write an email to my friends inviting them to dinner on Friday, it would be greatly appreciated.',
#   'input': '',
#   'output_1': "Dear Friends, \r\n\r\nI hope this message finds you well. I'm excited to invite you to dinner on Friday. We'll meet at 7:00 PM at [location]. I look forward to seeing you there. \r\n\r\nBest,\r\n[Name]",
#   'output_2': "Hey everyone! \n\nI'm hosting a dinner party this Friday night and I'd love for all of you to come over. We'll have a delicious spread of food and some great conversations. \n\nLet me know if you can make it - I'd love to see you all there!\n\nCheers,\n[Your Name]",
#   'annotator': 'davinci003_3',
#   'preference': 1.0}]
</code></pre>
</details>


<details>
<summary><b>Annotating list of output samples (not pairs)</b></summary>
```python
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_farm.utils import jload

# load some data
outputs_samples = jload("examples/data/multisamples_sft.json")[:3]
print(outputs_samples)
# ([{'instruction': 'Why is it important to save money?',)
# (  'input': '',)
# (  'output': 'Saving money is important because it allows you to have financial security in case of emergencies; to be able to afford the things you need; to have money for retirement; and to be able to afford the lifestyle you want. Saving money can also help you build wealth and reach your financial goals. It is important to develop the habit of regularly putting money aside for savings.'},)
# ( {'instruction': 'Why is it important to save money?',)
# (  'input': '',)
# (  'output': 'It is important to save money because it can help create a financial cushion in case of an emergency, allow you to pursue goals, and give you more financial security. Having a savings can also give you the opportunity to make more money by investing it. Saving money can also help you prepare for big expenditures, such as buying a car or a house.'},)
# ( {'instruction': 'Why is it important to save money?',)
# (  'input': '',)
# (  'output': 'Saving money is important because it provides financial security and allows for the purchase of large-ticket items, such as a car or house, or for emergency situations. It also helps to create financial independence and allows for more flexibility in life, such as taking unpaid time off from work or retiring early. Having money saved also provides peace of mind, knowing that one is prepared for any unexpected events.'}])

annotator = PairwiseAutoAnnotator()
annotated = annotator.annotate_samples(outputs_samples)

print(annotated)
[{'instruction': 'Why is it important to save money?',
#   'input': '',
#   'output_1': 'Saving money is important because it allows you to have financial security in case of emergencies; to be able to afford the things you need; to have money for retirement; and to be able to afford the lifestyle you want. Saving money can also help you build wealth and reach your financial goals. It is important to develop the habit of regularly putting money aside for savings.',
#   'output_2': 'Saving money is important because it provides financial security and allows for the purchase of large-ticket items, such as a car or house, or for emergency situations. It also helps to create financial independence and allows for more flexibility in life, such as taking unpaid time off from work or retiring early. Having money saved also provides peace of mind, knowing that one is prepared for any unexpected events.',
#   'annotator': 'davinci003_3',
#   'preference': 1.0}]
```
</details>

## Running Automatic Evaluation

## Running Reference Methods

We provide reference implementations of several methods for learning from pairwise feedback.
Example code to run these methods can be found in the `examples/` directory.
This includes [supervised fine-tuning](examples/supervised.py), [reward modeding](examples/reward_modeling.py)
, [RLHF with PPO](examples/rlhf_ppo.py), [best-of-n decoding](examples/best_of_n.py) and more.

Below we give example commands for reproducing the model artifacts in our paper. Notes:

- All training code are tested with FlashAttention enabled on a machine with 8 80GB A100 GPUs linked through NVLink.
- Best-of-n decoding was tested with a single 80GB GPU.
- Supervised fine-tuning and reward modeling can fit on 4 80GB A100 GPUs, while PPO training currently requires at least
  8
  80GB GPUs.
- Before running the code below, make sure to convert your LLaMA checkpoint and tokenizer into HuggingFace format and
  store it at `<your_path_to_hf_converted_llama_ckpt_and_tokenizer>`.

### Supervised fine-tuning (SFT)

To replicate our SFT10k model fine-tuned from LLaMA in the paper, run

```bash
bash examples/scripts/sft.sh \
  <your_output_dir_for_sft10k> \
  <your_wandb_run_name> \
  <your_path_to_hf_converted_llama_ckpt_and_tokenizer>
```

The SFT10k model will be saved at `<your_output_dir>`, and the name of the wandb run will be `<your_wandb_run_name>`.

### Reward modeling

To replicate our reward models trained in in the paper, run

```bash
bash examples/scripts/reward_modeling.sh \
  <your_output_dir_for_reward_model> \
  <your_wandb_run_name> \
  <your_output_dir_for_sft10k> \
  <preference_dataset_name>
```

Set `<preference_dataset_name>` to `"alpaca_noisy_multi_preference"` for simulated preference reward model, and
`"alpaca_human_preference"` for human preference reward model.

### RLHF with PPO

To replicate our RLHF PPO model trained with simulated reward model in the paper, run

```bash
bash examples/scripts/rlhf_ppo.sh \
  <your_output_dir> \
  <your_wandb_run_name> \
  <your_output_dir_for_reward_model> \
  <your_output_dir_for_sft10k> \
  <kl_coef>
```

`<your_output_dir_for_reward_model>` should point to either simulated reward model or human reward model trained according
to the previous step.
Note the KL penalty coefficient for human reward PPO is much larger than for simulated PPO.
Set `<kl_coef>` to `0.0067` for simulated PPO, and `0.0002` for human PPO to recover our original results.
Performance of the PPO model is typically much better than SFT at 20-80 PPO steps (less than 4 pass through the entire
set of instructions), and starts to decay with more PPO steps.

### Best-of-n decoding

To replicate our best-of-n inference-time decoding results for the AlpacaFarm evaluation suite, run

```bash
python examples/best_of_n.py \
  --task "run_best_of_n" \
  --decoder_name_or_path <your_output_dir_for_decoder> \  # Can be SFT model or even PPO tuned model.
  --scorer_name_or_path <your_output_dir_for_reward_model> \
  --num_return_sequences 16 \  # This is the n in best-of-n.
  --per_device_batch_size 4 \  # Reduce this if you don't have enough memory.
  --split "eval" \
  --mixed_precision "bf16" \
  --tf32 True \
  --flash_attn True \
  --output_path <your_output_path_to_store_samples>
```

You can then use the generated samples at `<your_output_path_to_store_samples>` directly with our automated evaluation.

## Citation

Please consider citing our work if you use the data or code in this repo.

```
@misc{alpaca,
  author = {Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani and Jimmy Ba and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback},
  year = {2023},
  howpublished = {\url{https://tatsu-lab.github.io/alpaca_farm_paper.pdf}},
}
```
