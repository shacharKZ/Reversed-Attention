# Reversed Attention

## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)

## Description
This repo contains the official code for the paper: Reversed Attention: On The Gradient Descent Of Attention Layers in GPT (NAACL 2025)

Please try our demo: [![Colab Reversed Attention](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13HDQ6o-TN7PcKCk4DlKgQ9O6jEeHbUW0?usp=sharing)

![RA overview](https://github.com/user-attachments/assets/49b98a34-be12-4e54-88ce-9c4879faceb5)


## Installation

### Prerequisites
Make sure you have Python 3.10 installed on your machine.

All files must be in the same folder

### Installing Dependencies
In your Python environment, install the required dependencies using `pip` and the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

Make sure all files are in the same folder and run `main_demo.ipynb`.

This script demonstrates how to obtain the backward pass attention, which we named Reversed Attention.

This demo can be run in Colab and does not require a GPU (but it can make the run faster)


### Full Experiments
The full experiments code can be found in [`src`](src).

The scripts are provided as notebooks (.ipynb). If you want to run them as simple python scripts, convert them into .py files using:
```bash
jupyter nbconvert <script_name>.ipynb --to python
```

Some functionality are provided by the submodule `function_vectors' (make sure you pull it and follow its setup instruction).

In particular, the data for the experiments is sourced from that submodule. Part of the datasets need to be manually downloaded from [here](https://lre.baulab.info/data/) (credit for this datasets creator at [Linearity of Relation Decoding in Transformer LMs](https://lre.baulab.info/)). For more information see the two scripts under [`function_vectors/dataset_files'](function_vectors/dataset_files).


## Citing

```bibtex
@article{katz2024rAttention,
  title={Reversed Attention: On The Gradient Descent Of Attention Layers In GPT},
  author={Shahar Katz and Lior Wolf},
  journal={arXiv preprint arXiv:2412.17019},
  year={2024}
}
```
