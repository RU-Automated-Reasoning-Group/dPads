# Differentiable Synthesis of Program Architecture
---

## Introduction
We propose a differentiable architecture search method for program synthesis, including top-$N$ preservation and variant of A$^*$ search. This README file is created for execution guidance, including requirement to run the code and how to implement experiment. Besides, we also show some experiment results here.

## Requirements
- Python 3.6+
- PyTorch 1.4.0+
- scikit-learn 0.22.1+
- Numpy
- tqdm

## Code Structures
- `train_nas.py` includes main function to start program architecture search with program derivation graph. It also create necessary parameter list for program derivation graph class and call functions to start the search.
- `eval_test.py` is created for evaluation. Given synthesized program, this file could be called to test F1 score, accuracy on defined dataset.
- `dsl_*.py` are files defining specific parameter functions to extract feature for related dataset.
- `dsl/` contains common grammars in our DSL to synthesize a program.
- `algorithms/` contains the main file `nas.py` to search program architecture with top-$N$ preservation and A$^*$ search.
- `program_graph.py` defines program derivation graph, including nodes, edges and graph structure.
- `utils/` contains useful codes for dataloader, loss function, evaluation and creating logs file.

## Run the Code
#### Program Architecture Search
To synthesize program with dPads, use `train_nas*.sh` files with command `sh train_nas*.sh`. 
- To search for program on Crim13 dataset, please run `sh train_nas.sh`; 
- To search program on Fly-vs-fly dataset, please run `sh train_nas_fly.sh`; 
- To search program on Basketball dataset, please run `sh train_nas_basketball.sh`; 
- To search program on SK152 dataset, please run `sh train_nas_sk152.sh`.

#### Evaluation
To evaluate sythesized program on test set, use `sh eval_test.sh`.

#### Implement Code for NEAR

To run experiment of NEAR, please download the code and follow the instruction [here](https://github.com/trishullab/near). NEAR contains similar argument as ours and our released datasets can be directly used in code of NEAR.

## Data

All the data we use to synthesize program can be downloaded from [google drive](https://drive.google.com/drive/folders/1NWn1VXJKk1GowsDOZfzcnVR5vd46u5Jy?usp=sharing).
#### Crim13 dataset
To search program on Crim13 dataset, please refer to [NEAR](https://github.com/trishullab/near/tree/master/near_code) github to download related dataset and save the `crim13_processed` directory in path `data/`. Users could also modify command-line arguments `--train_data`, `--valid_data`, `--test_data`, `--train_labels`, `--valid_labels` and `--test_labels` to custom path of data.

#### Fly-vs-fly dataset
The original dataset could be downloaded [here](https://data.caltech.edu/records/1893), which includes videos and extracted feature trajectory. Processed dataset used in our experiment are contained in `fly_process` directory of google drive.

#### Basketball dataset
The original of Basketball dataset could be download in [aws](https://aws.amazon.com/marketplace/pp/prodview-7kigo63d3iln2?qid=1606330770194&sr=0-1&ref_=srh_res_product_title#offers). User could download processed dataset in `basketball_processed` directory of google drive.

#### Skeletics-152 dataset
The original Skeletic-152 dataset could be accessed from [here](https://github.com/skelemoa/quovadis/tree/master/skeletics-152). The process dataset for our experiment could be downloaded in `sk152_process`

## More Details on Command-line Arguments

To specific details of dataset, `--input_type` and `--output_type` illustrate whether input/output data is "list" or "atom". "list" refers to sequence data and "atom" refers to value. `--input_size` and `--output_size` indicate the dimension of each atom in input/output data, `--num_label` indicates number of categories in the task.

For training setting, `--lossfxn` refers to specific kind of loss function applied to train derivation graph, choices include "crossentropy", "bcelogits" and "softf1". `--max_depth` indicates the depth limitation of derivation graph, `--symbolic_epochs` refers to number of epochs for iterative training, `--neural_epochs` refers to number of epochs during finetuning. `--train_valid_split` defines the ratio to split train and valid data for iterative training, and `--batch_size` refers to number of data in a batch for entire training and searching process.

Moreover, as mentioned in our paper, node-sharing and graph iterative unfolding are leveraged to reduce complexity of dPads, as specific by `--node_share` and `--graph_unfold`. User can reproduce ablation study results in the paper by de-activing either argument.

## Experiment Results
We pick 5 random seeds (0,1000,2000,3000,4000) and run dPads to synthsize program architectures on four datasets. Propose dPads achieve current state-of-the-art results.

||Crim13|Fly-vs-fly|Basketball|SK152|
|:---|:---:|:---:|:---:|:---:|
|**F1**|0.458|0.887|0.945|0.337|
|**Accuracy**|0.812|0.853|0.939|0.337|
|**Time (min)**|147.87|348.25|174.68|162.70|