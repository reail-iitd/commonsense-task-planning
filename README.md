# Using Commonsense Generalization for Predicting Tool Use for Robot Plan Synthesis

This implementation contains all the models mentioned in the paper for next-tool prediction along with next-action prediction.

## Contents 
- Introduction
- Environment Setup
- Directory Structure
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
(commonsense_tool) $ git clone https://github.com/reail-iitd/commonsense-task-planning.git
(commonsense_tool) $ cd Robot-task-planning
(commonsense_tool) $ git checkout release
(commonsense_tool) $ pip3 install -r requirements.txt
```

## Directory Structure
| Folder/File                       | Utility 		              
| --------------------------------- | --------------------------- 
| app.py 				            | This is the main file to run the data collection platform.
|									  This file starts the web server at the default port 5000 on localhost.
|									  It starts an instance of PyBullet on the system and exposes the simulator
|									| to the user at the appropriate website.
| train.py                            This is the main file used to train and evaluate all the 
|									| models as mentioned in the paper.
| husky_ur5.py                        This is the main file for the PyBullet simulator. It is responsible
|									  for loading the PyBullet simulator, running the appropriate action
|									  and sending the appropriate exceptions wherever applicable. It
|									  also checks if the goal specified has been completed.
| src/GNN/CONSTANTS.py				| This file contains the global constant used by the training
|									  pipeline like the number of epochs, hidden size used etc.
| src/GNN/dataset_utils.py          | This file contains the Dataset class, which can be used to process
|									  the dataset in any form as required by the training pipeline.
| src/GNN/\*models 	                | These contain the different PyTorch models that were used for training
|									  the system.
| src/datapoint.py 				    | This contains the datapoint class. All datapoints found in the dataset are
|									  an instance of this class.
| jsons/embeddings                  | These contain the files corresponding to
|									  [fasttext](https://fasttext.cc/docs/en/english-vectors.html) and [conceptnet] 
|									  (https://github.com/commonsense/conceptnet-numberbatch) embeddings.
| jsons/\*\_goals                   | These contain the goals which can be completed by the robot
|									  in the factory and the home domain.
| jsons/\*\_worlds      	        | These contain the different world instances in the home
|									  and factory domain.
| jsons/\*.json 					| These are configuration files for the simulator and the webapp.
|									  These define the different actions possible in the simulator,
|									  the different objects present in the simulator, states possible, readable
|									  form of actions to show on the webapp etc.
| models/\*                         | This folder contains the stl and urdf files for all models 
|									  which are loaded by the physics simulator used i.e Pybullet.
| templates/*						| These are the templates that are used by the webapp to load the
|									  different tutorial webpages along with the actual data collection
|									  platform.

## Setup Web Interface for Data Collection
To execute the website that is needed for data collection, use the following command:
```bash
$ python3 app.py --randomize
```
The website can now be accessed by using the link (http://0.0.0.0:5000/).
For all the arguments that can be provided look at the help provided,

```bash
$ python3 app.py --help
```

## Dataset

Download the dataset [here](https://drive.google.com/open?id=18dmWjDjz3DPYZTFv92vAnMssK2YFZh3j).

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

All the trained models can be found [here](https://drive.google.com/open?id=1Kw65B55DehnteO1hwLUk0k1rWw2eCTl0). These can be run by setting
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


