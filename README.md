# Differentiable Program Synthesis
---

## Introduction
We propose a differentiable approach for synthesizing differentiable programs [paper](https://proceedings.neurips.cc/paper/2021/file/5c5a93a042235058b1ef7b0ac1e11b67-Paper.pdf).

## Requirements
- Python 3.6+
- PyTorch 1.4.0+
- scikit-learn 0.22.1+
- Numpy
- tqdm

## Code Structures
- `train_nas.py` includes the main function to synthesize differentiable programs.
- `eval_test.py` is used for evaluation. Given a synthesized program, it calculates the program's F1 score and accuracy on a dataset.
- `dsl_*.py` defines the parameterized domain-specific languages for the benchmarks.
- `dsl/` includes the semantics of the predefined domain-specific languages.
- `algorithms/` contains the main synthesis algorithm (`nas.py`).
- `program_graph.py` defines our algorithm's learning representation.
- `utils/` includes useful code for data loading, loss functions, etc.

## Run the Code
#### Program Search
- To search a programmatic classifier for the Crim13 dataset, run `sh train_nas.sh`; 
- To search a programmatic classifier for the Fly-vs-fly dataset, run `sh train_nas_fly.sh`; 
- To search a programmatic classifier for the Basketball dataset, run `sh train_nas_basketball.sh`; 
- To search a programmatic classifier for the SK152 dataset, run `sh train_nas_sk152.sh`.

#### Program Evaluation
To evaluate a synthesized program on its test dataset, use `sh eval_test.sh`.

#### Comparision against NEAR

To run NEAR with our released datasets, please download its source code and follow the instructions from [here](https://github.com/trishullab/near).

## Data

All the datasets can be downloaded from [here](https://drive.google.com/drive/folders/1NWn1VXJKk1GowsDOZfzcnVR5vd46u5Jy?usp=sharing).

## More Details about dPads's Command-line Arguments for Sequence Classification

`--input_type` and `--output_type` defines if the program input/output data is over "list" or "atom". "list" refers to sequences of data and "atom" refers to values. 

`--input_size` and `--output_size` defines the dimension of each frame in an input (output) sequence.

`--num_label` defines the number of classification categories for a sequence classification task.

`--lossfxn` defines the loss function for training. The choices include "crossentropy", "bcelogits" and "softf1". 

`--max_depth` defines the maximum depth of the abstract syntax tree of any synthesized program. 

`--symbolic_epochs` defines the number of epochs used for program search on top of a program derivation graph.

`--neural_epochs` defines the number of epochs used for program selection from a trained program derivation graph. 

`--train_valid_split` defines the ratio to split the training dataset for synthesizing program structures and program parameters.

`--batch_size` defines the batch size for training.

`--node_share` and `--graph_unfold`. Please refer to the ablation study section of our paper.

## Experiment Results
We report dPad's results on the four datesets, averaged over 5 random seeds (0,1000,2000,3000,4000):

||Crim13|Fly-vs-fly|Basketball|SK152|
|:---|:---:|:---:|:---:|:---:|
|**F1 Scores**|0.458|0.887|0.945|0.337|
|**Accuracy**|0.812|0.853|0.939|0.337|
|**Time (mins)**|147.87|348.25|174.68|162.70|
