<p align="center" width="100%">
<a href="https://crfm.stanford.edu/alpaca/" target="_blank"><img src="assets/AlpacaFarm_big.png" alt="AlpacaFarm" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# AlpacaFarm: A Simulation Framework for <br/>Methods that Learn from Human Feedback

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Research and development on learning from human feedback is difficult because methods like [RLHF](https://arxiv.org/abs/2203.02155) are costly to run and complex to analyze.
AlpacaFarm is a simulator that enables research and development on learning from feedback at a fraction of the usual cost,
promoting accessible research on instruction following and alignment.

This repo contains code for

- [simulating preference feedback from language models such as GPT-4](#simulating-pairwise-preference)
- [automated evaluation for instruction-following models](#running-automatic-evaluation)
- [validated reference implementations of baseline methods such as PPO and best-of-n](#running-reference-methods)

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
low-cost pairwise feedback generators, automated evaluations for methods development, and reference implementations of learning algorithms for comparison and modification. 

To reduce annotation cost, we design prompts for API LLMs (e.g. GPT-4, ChatGPT) that enable us to simulate human feedback for 45x cheaper than crowdworkers.

For the challenge of evaluation, we use user interactions with the Alpaca Demo as guidance and mimic this distribution by combining several existing public evaluation datasets including the self-instruct eval set, the anthropic helpful evaluation, the open assistant evaluation, Koala evaluation, and Vicuna evaluation.
On top of this evaluation distribution, we adopt pairwise evaluation as our protocol and measure the win-rate against Davinci003.

We implement and test several popular learning algorithms (e.g. PPO, expert iteration, best-of-n sampling), and release their implementations as resources.

With this design, we show that our simulation is accurate. When we train and develop methods in simulation, the rankings of these methods agree closely with what we see when we train and develop the same methods using actual human feedback.
<p style="text-align:center;">
  <img style="max-width:70%; height:auto;" src="./assets/method_corr.png" alt="Workflow">
</p>

## Installation

For basic installation, run

```bash
pip install git+https://github.com/tatsu-lab/alpaca_farm.git
```

To enable FlashAttention and other optimizations, install
the [`flash-attn`](https://github.com/HazyResearch/flash-attention) and [`apex`](https://github.com/NVIDIA/apex)
packages.

## Simulating Pairwise Preference

## Running Automatic Evaluation

## Running Reference Methods

### Citation

Please consider citing our work if you use the data or code in this repo.

```
TODO
```
