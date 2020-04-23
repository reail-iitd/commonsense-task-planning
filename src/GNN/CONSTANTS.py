import json
import numpy as np
import torch
from copy import deepcopy
from sys import argv
# Contains all the constants used by the model. These are global constants.

domain = argv[1] # can be 'home' or 'factory'

# The embedding type which is used by model. Can be conceptnet or fasttext.
embedding = "conceptnet" if (("_C" in argv[3]) ^ ("Final" in argv[3])) or "Action" in argv[3] else "fasttext"

# These are the states that are possible for any object. Only the ones possessed bu the object are 1. Other are 0.
STATES = ["Outside", "Inside", "On", "Off", "Close", "Open", "Up", "Down", "Sticky", "Non_Sticky", "Dirty", "Clean", "Grabbed", "Free", "Welded", "Not_Welded", "Drilled", "Not_Drilled", "Driven", "Not_Driven", "Fueled", "Not_Fueled", "Cut", "Not_Cut", "Painted", "Not_Painted", "Different_Height", "Same_Height"]
if domain == 'factory': STATES += ['To_Print', "Printed"]

N_STATES = len(STATES)
state2indx = {}
for i,state in enumerate(STATES):
	state2indx[state] = i

# All the edges that are possible in the graph.
EDGES = ["Close", "Inside", "On", "Stuck"]
N_EDGES = len(EDGES)
edge2idx = {}
for i,edge in enumerate(EDGES):
	edge2idx[edge] = i
NUMOBJECTS = len(json.load(open("jsons/objects.json", "r"))["objects"])

PRETRAINED_VECTOR_SIZE = 300
REDUCED_DIMENSION_SIZE = 64
SIZE_AND_POS_SIZE = 10

# EMBEDDING_DIM = 32
N_TIMESEPS = 2
GRAPH_HIDDEN = 64
NUM_EPOCHS = 100
LOGIT_HIDDEN = 32
NUM_GOALS = 8

TOOLS2 = ['stool', 'tray', 'tray2', 'big-tray', 'book', 'box', 'chair',\
		'stick', 'glue', 'tape', 'mop', 'sponge', 'vacuum'] if domain == 'home' else ['lift', \
		'stool', 'trolley', 'stick', 'ladder', 'glue', 'tape', 'drill', '3d_printer', \
		'screwdriver', 'brick', 'hammer', 'blow_dryer', 'box', 'wood_cutter', 'welder', \
		'spraypaint', 'toolbox', 'mop']
TOOLS = TOOLS2 + ['no-tool']
NUMTOOLS = len(TOOLS)
MODEL_SAVE_PATH = "trained_models/"
AUGMENTATION = 1

goal_jsons = ["jsons/home_goals/goal1-milk-fridge.json", "jsons/home_goals/goal2-fruits-cupboard.json",\
            "jsons/home_goals/goal3-clean-dirt.json", "jsons/home_goals/goal4-stick-paper.json",\
            "jsons/home_goals/goal5-cubes-box.json", "jsons/home_goals/goal6-bottles-dumpster.json",\
            "jsons/home_goals/goal7-weight-paper.json", "jsons/home_goals/goal8-light-off.json"] if domain == 'home' else [\
            "jsons/factory_goals/goal1-crates-platform.json", "jsons/factory_goals/goal2-paper-wall.json",\
            "jsons/factory_goals/goal3-board-wall.json", "jsons/factory_goals/goal4-generator-on.json", \
            "jsons/factory_goals/goal5-assemble-parts.json", "jsons/factory_goals/goal6-tools-workbench.json", \
            "jsons/factory_goals/goal7-clean-water.json", "jsons/factory_goals/goal8-clean-oil.json"]
goalObjects = {}
for i in range(len(goal_jsons)):
	goal_json = json.load(open(goal_jsons[i], "r"))
	goalObjects[i+1] = goal_json["goal-objects"]

object2idx = {};
for i, obj in enumerate(json.load(open("jsons/objects.json", "r"))["objects"]):
	object2idx[obj["name"]] = i

def compute_constants(embedding):
	# Global Constants
	embeddings = None
	with open('jsons/embeddings/' + embedding + '.vectors') as handle:
		embeddings = json.load(handle)

	# Object to vectors
	object2vec = {}; idx2object = {}
	for i, obj in enumerate(json.load(open("jsons/objects.json", "r"))["objects"]):
		object2vec[obj["name"]] = embeddings[obj["name"]]
		idx2object[i] = obj["name"]
	tool_vec = torch.Tensor([object2vec[i] for i in TOOLS2])

	# Goal objects and vectors
	goal2vec, goalObjects2vec = {}, {}
	for i in range(len(goal_jsons)):
		goal_json = json.load(open(goal_jsons[i], "r"))
		goal2vec[i+1] = torch.Tensor(np.array(goal_json["goal-vector"]))
		if embedding == 'conceptnet' and "goal-vector-conceptnet" in goal_json:
			goal2vec[i+1] = torch.Tensor(np.array(goal_json["goal-vector-conceptnet"]))
		goal_object_vec = np.zeros(300)
		for j in goal_json["goal-objects"]:
			goal_object_vec += object2vec[j]
		goal_object_vec /= len(goal_json["goal-objects"])
		goalObjects2vec[i+1] = torch.Tensor(goal_object_vec)
	return embeddings, object2vec, object2idx, idx2object, tool_vec, goal2vec, goalObjects2vec

embeddings, object2vec, object2idx, idx2object, tool_vec, goal2vec, goalObjects2vec = compute_constants(embedding)