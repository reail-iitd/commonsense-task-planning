import json
from src.GNN.CONSTANTS import *
import numpy as np
import os
import pickle
from src.datapoint import Datapoint
import dgl
import torch
from tqdm import tqdm
import math
from scipy.spatial import distance
from sys import maxsize

etypes = ["Close", "Inside", "On", "Stuck"]

############################ DGL ############################

def getToolSequence(actionSeq):
	"""Returns the sequence of tools that were used in the plan."""
	toolSeq = ['no-tool'] * len(actionSeq)
	currentTool = 'no-tool'
	for i in range(len(toolSeq)-1, -1, -1):
		for obj in actionSeq[i]['args']:
			if obj in TOOLS2: 
				currentTool = obj
				break
		toolSeq[i] = currentTool
	return toolSeq

def getGlobalID(dp):
	maxID = 0
	for i in dp.metrics[0].keys():
		maxID = max(maxID, object2idx[i])
	return maxID + 1

def convertToDGLGraph(graph_data, globalNode, goal_num, globalID):
	""" Converts the graph from the datapoint into a DGL form of graph."""
	# Make edge sets
	close, inside, on, stuck = [], [], [], []
	for edge in graph_data["edges"]:
		if edge["relation"] == "Close": close.append((edge["from"], edge["to"]))
		elif edge["relation"] == "Inside": inside.append((edge["from"], edge["to"]))
		elif edge["relation"] == "On": on.append((edge["from"], edge["to"]))
		elif edge["relation"] == "Stuck": stuck.append((edge["from"], edge["to"]))
	edgeDict = {
		('object', 'Close', 'object'): close,
		('object', 'Inside', 'object'): inside,
		('object', 'On', 'object'): on,
		('object', 'Stuck', 'object'): stuck
		}
	if globalNode:
		globalList = []
		for i in range(globalID): globalList.append((i, globalID))
		edgeDict[('object', 'Global', 'object')] = globalList
	g = dgl.heterograph(edgeDict)
	# Add node features
	n_nodes = g.number_of_nodes()
	node_states = torch.zeros([n_nodes, N_STATES], dtype=torch.float) # State vector
	node_vectors = torch.zeros([n_nodes, PRETRAINED_VECTOR_SIZE], dtype=torch.float) # Fasttext embedding 
	node_size_and_pos = torch.zeros([n_nodes, 10], dtype=torch.float) # Size and position
	node_in_goal = torch.zeros([n_nodes, 1], dtype=torch.float) # Object in goal
	for i, node in enumerate(graph_data["nodes"]):
		states = node["states"]
		node_id = node["id"]
		for state in states:
			idx = state2indx[state]
			node_states[node_id, idx] = 1
		node_vectors[node_id] = torch.FloatTensor(node["vector"])
		node_size_and_pos[node_id] = torch.FloatTensor(list(node["size"]) + list(node["position"][0]) + (list(node["position"][1]) if len(node['position'][1]) > 0 else [0, 0, 0, 0]))
		node_in_goal[node_id] = 1 if node["name"] in goalObjects[goal_num] else 0

	g.ndata['feat'] = torch.cat((node_vectors, node_states, node_size_and_pos), 1)
	return g

def getDGLGraph(pathToDatapoint, globalNode, ignoreNoTool, e):
	""" Returns the intital state DGL graph from the path to the given datapoint."""
	datapoint = pickle.load(open(pathToDatapoint, "rb"))
	time = datapoint.totalTime()
	tools = datapoint.getTools(not ignoreNoTool)
	if ignoreNoTool and len(tools) == 0: return None
	goal_num = int(datapoint.goal[4])
	world_num = int(datapoint.world[-1])
	# Initial Graph
	graph_data = datapoint.getGraph(embeddings = e)["graph_0"] 
	g = convertToDGLGraph(graph_data, globalNode, goal_num, getGlobalID(datapoint) if globalNode else -1)
	return (goal_num, world_num, tools, g, time)

def getDGLSequence(pathToDatapoint, globalNode, ignoreNoTool, e):
	""" Returns the entire sequence of graphs and actions from the plan in the provided datapoint."""
	datapoint = pickle.load(open(pathToDatapoint, "rb"))
	time = datapoint.totalTime()
	tools = datapoint.getTools(not ignoreNoTool)
	if ignoreNoTool and len(tools) == 0: return None
	goal_num = int(datapoint.goal[4])
	world_num = int(datapoint.world[-1])
	actionSeq = []; graphSeq = []
	for action in datapoint.symbolicActions:
		if not (str(action[0]) == 'E' or str(action[0]) == 'U'): actionSeq.append(action[0])
	for i in range(len(datapoint.metrics)):
		if datapoint.actions[i] == 'Start': graphSeq.append(convertToDGLGraph(datapoint.getGraph(i, embeddings=e)["graph_"+str(i)], globalNode, goal_num, getGlobalID(datapoint) if globalNode else -1))
	assert len(actionSeq) == len(graphSeq)
	toolSeq = getToolSequence(actionSeq)
	return (goal_num, world_num, toolSeq, (actionSeq, graphSeq), time)

class DGLDataset():
	""" Class which contains the entire dataset.
		For any i,
		self.graphs[i][-2] -> Every element of this object is a datapoint.
					  If sequence is true this is a graph sequence, otherwise it contains the initial state of the datapoint.

		self.graphs[i][-3] -> The tools used in the datapoint. If sequence is true contains the sequence of next most recent tool. 
						Otherwise, if sequence is false, contains a list of tools used in the plan.

	"""
	def __init__(self, program_dir, augmentation=50, globalNode=False, ignoreNoTool=False, sequence=False, embedding="conceptnet"):
		global etypes
		if globalNode: etypes.append('Global')
		graphs = []
		with open('jsons/embeddings/'+embedding+'.vectors') as handle: e = json.load(handle)
		self.goal_scene_to_tools = {}; self.min_time = {}
		all_files = list(os.walk(program_dir))
		for path, dirs, files in tqdm(all_files):
			if (len(files) > 0):
				for file in files:
					file_path = path + "/" + file
					for i in range(augmentation):
						graph = getDGLGraph(file_path, globalNode, ignoreNoTool, e) if not sequence else getDGLSequence(file_path, globalNode, ignoreNoTool, e)
						if graph: 
							graphs.append(graph)
							tools = graphs[-1][2]
							goal_num = graphs[-1][0]
							world_num = graphs[-1][1]
							if (goal_num,world_num) not in self.goal_scene_to_tools:
								self.goal_scene_to_tools[(goal_num,world_num)] = []
								self.min_time[(goal_num,world_num)] = maxsize
							for tool in tools:
								if tool not in self.goal_scene_to_tools[(goal_num,world_num)]:
									self.goal_scene_to_tools[(goal_num,world_num)].append(tool)
							self.min_time[(goal_num,world_num)] = min(self.min_time[(goal_num,world_num)], graphs[-1][4])
		self.graphs = graphs
		self.features = self.graphs[0][3].ndata['feat'].shape[1] if not sequence else self.graphs[0][3][1][0].ndata['feat'].shape[1]
		self.num_objects = self.graphs[0][3].number_of_nodes() if not sequence else self.graphs[0][3][1][0].number_of_nodes()
		if globalNode: self.num_objects -= 1

class TestDataset():
	def __init__(self, program_dir, augmentation=1):
		graphs = []
		all_files = os.walk(program_dir)
		for path, dirs, files in tqdm(all_files):
			if (len(files) > 0):
				for file in files:
					file_path = path + "/" + file
					for i in range(augmentation):
						with open(file_path, 'r') as handle:
						    graph = json.load(handle)
						g = convertToDGLGraph(graph["graph_0"], False, graph["goal_num"], -1)	
						graphs.append((graph["goal_num"], int(path[-1]), graph["tools"], convertToDGLGraph(graph["graph_0"], False, graph["goal_num"], -1), graph["tool_embeddings"]))					
		self.graphs = graphs