# Using Commonsense Generalization for Predicting Tool Use for Robot Plan Synthesis

This implementation contains all the models mentioned in the paper for next-tool prediction along with next-action prediction.

## Contents 
- Introduction
- Environment Setup
- Setup Web Interface for Data Collection
- Dataset
- Training models

## Introduction

<img src="/screenshot.png" width="900" align="middle">

In order to effectively perform activities in realistic environments like a home or a factory, robots often require tools. Determining if a tool could be effective in accomplishing a task can be helpful in scoping the search for feasible plans. This work aims at learning such "commonsense" to generalize and adapt to novel settings where one or more known tools are missing with the presence of unseen tools and world scenes. Towards this objective, we crowd-source a data set of humans instructing a robot in a Physics simulator perform tasks involving multi-step object interactions with symbolic state changes. The data set is used to supervise a learner that predicts the use of an object as a tool in a plan achieving the agent’s goal. The model encodes the agent’s environment, including metric and semantic properties, using gated graph convolutions and incorporates goal- conditioned spatial attention to predict the optimal tool to use. We demonstrate generalization for predicting tool use for objects unseen in training data by conditioning the model on pre-trained embeddings derived from a relational knowledge source such as ConceptNet. Experiments show that the learned model can accurately predict common use of tools based on spatial context, semantic attribute of objects and goals specifications and effectively generalize to novel scenarios with world instances not encountered in training.

## Environment Setup

```bash
$ python3 -m venv commonsense_tool
$ source commonsense_tool/bin/activate
(commonsense_tool) $ git clone https://github.com/shreshthtuli/Robot-task-planning.git
(commonsense_tool) $ cd Robot-task-planning
(commonsense_tool) $ git checkout release
(commonsense_tool) $ pip3 install -r requirements.txt
```

## Setup Web Interface for Data Collection
To execute the website that is needed for data collection, use the following command:
```bash
$ python3 app.py --world WORLD --randomize
```
WORLD here can be home/factory.
The website can now be accessed by using the link (http://0.0.0.0:5000/).
For all the arguments that can be provided look at the help provided,

```bash
$ python3 app.py --help
```

## Dataset

Download the program dataset [here](https://drive.google.com/file/d/1txrMTiVnhxBhblf6ypGJ_m3jr3MBBlUe/view?usp=sharing).

Here is how the dataset structure should look like:

```
dataset
├── home
├── factory
└── test
    ├── home
    └── factory
```

## Training

All the trained models can be found [here](http://xyz/temp_placeholder). These can be run by setting
train variable to false in `train.py`.

All the models mentioned in the paper can be trained through the command

```bash
$ python3 train.py DOMAIN EMBEDDING MODEL_TYPE
```
Here DOMAIN can be home/factory.
EMBEDDING is the embeddings that will be used by the model
and can be conceptnet/fasttext.
All ablated models can also be trained by changing the model variable appropriately and can be found in `src/GNN/models.py` and `src/GNN/action_models.py`. The following table
contains what the MODEL_TYPE can be along with its name as mentioned in the paper.

| MODEL_TYPE                        | Name of model (as in paper) |
| --------------------------------- | --------------------------- |
| agcn-likelihood                   | GGCN+Metric+Attn+L+NT+C+W   |
| sequence_baseline_metric_att_aseq | GGCN+Metric+Attn+Aseq       |


