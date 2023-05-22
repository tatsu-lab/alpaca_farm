# Stanford AlpacaFarm: A Simulation Framework for <br/>Methods that Learn from Human Feedback

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Research and development on instruction-following models with human feedback is difficult because methods that learn
from human feedback
are costly and complex.
Stanford AlpacaFarm is a simulator that enables research and development for learning from feedback at a low cost.

This repo contains code for

- simulating preference feedback from language models
- automated evaluation of instruction-following models
- reference implementations of baseline methods

<br>
<p style="text-align:center;">
  <img style="max-width:80%; height:auto;" src="./assets/fig1.jpg" alt="Workflow">
</p>

**Usage and License Notices**: Alpaca is intended and licensed for research use only. The dataset is CC BY NC 4.0 (
allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
The weight diff is also CC BY NC 4.0 (allowing only non-commercial use).

## Overall
TODO: this part should detail the model development process. the three components.

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
