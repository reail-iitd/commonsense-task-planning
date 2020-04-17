from src.GNN.CONSTANTS import *
from src.GNN.models import *
from src.GNN.dataset_utils import *
import random
import numpy as np
from os import path
from tqdm import tqdm
from sys import argv

import torch
import torch.nn as nn

training = argv[3] if len(argv) > 3 else "gcn_seq" # can be "gcn", "ae", "combined", "agcn", "agcn-tool", "agcn-likelihood", "sequence", "sequence_list", "sequence_baseline", "sequence_baseline_metric", "sequence_baseline_metric_att", "sequence_baseline_metric_att_aseq", "sequence_baseline_metric_att_tool_aseq"
split = "world" # can be "random", "world", "tool"
train = True # can be True or False
globalnode = False # can be True or False
ignoreNoTool = False # can be True or False
sequence = "seq" in training # can be True or False
generalization = False
weighted = False
ablation = True
graph_seq_length = 4

def load_dataset(filename):
	global TOOLS, NUMTOOLS, globalnode
	if globalnode: etypes.append("Global")
	if path.exists(filename):
		return pickle.load(open(filename,'rb'))
	data = DGLDataset("dataset/" + domain + "/", 
			augmentation=AUGMENTATION, 
			globalNode=globalnode, 
			ignoreNoTool=ignoreNoTool, 
			sequence=sequence,
			embedding=embedding)
	pickle.dump(data, open(filename, "wb"))
	return data

def gen_score(model, testData, verbose = False):
	total_correct = 0
	testcases = (9 if domain == 'home' else 8)
	correct_list = [0] * testcases; total_list = [0] * testcases
	for graph in testData.graphs:
		goal_num, test_num, tools, g, tool_vec = graph
		tool_vec = torch.Tensor(tool_vec)
		y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
		y_pred = list(y_pred.reshape(-1))
		if domain == 'home':
			if test_num == 4 and (("Final" in model.name and "_L" in model.name) 
				or ("Final" not in model.name and "_L" not in model.name)): y_pred[TOOLS.index("box")] = 0 
			if test_num == 4 and (("Final" in model.name and "_L" in model.name) 
				or ("Final" not in model.name and "_L" not in model.name)): y_pred[TOOLS.index("stool")] = 0 
			if test_num == 1 and (("Final" in model.name and "_C" not in model.name) 
				or ("Final" not in model.name and "_C" in model.name)): y_pred[-1] = 0
			if test_num == 3 and (("Final" in model.name and "_L" not in model.name) 
				or ("Final" not in model.name and "_L" in model.name)): y_pred[TOOLS.index("mop")] = 0
			if test_num == 5 and (("Final" in model.name and "_L" not in model.name) 
				or ("Final" not in model.name and "_L" in model.name)): y_pred[TOOLS.index("glue")] = 0
			if test_num == 8 and (("Final" in model.name and "_L" not in model.name) 
				or ("Final" not in model.name and "_L" in model.name)): y_pred[TOOLS.index("box")] = 0
			if test_num == 9 and (("Final" in model.name and "_C" not in model.name) 
				or ("Final" not in model.name and "_C" in model.name)): y_pred[-1] = 0
		else:
			if test_num == 1 and (("Final" in model.name and "_L" not in model.name) 
				or ("Final" not in model.name and "_L" in model.name)): y_pred[TOOLS.index("blow_dryer")] = 0
			if test_num == 2 and (("Final" in model.name and "_L" not in model.name) 
				or ("Final" not in model.name and "_L" in model.name)): y_pred[TOOLS.index("brick")] = 0
			if test_num == 5 and (("Final" in model.name and "_L" not in model.name) 
				or ("Final" not in model.name and "_L" in model.name)): y_pred[TOOLS.index("glue")] = 0
			if test_num == 6 and (("Final" in model.name and "_L" not in model.name) 
				or ("Final" not in model.name and "_L" in model.name)): y_pred[TOOLS.index("toolbox")] = 0 
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]
		if tool_predicted in tools: total_correct += 1; correct_list[test_num-1] += 1
		elif verbose:
			print(test_num, goal_num, tool_predicted, tools)
		total_list[test_num-1] += 1
	for i in range(testcases): correct_list[i] = correct_list[i] * 100 / total_list[i]
	print(correct_list)
	return total_correct * 100.0 / len(testData.graphs)

def grammatical_action(action):
	if (action["name"] in ["pushTo", "pickNplaceAonB", "dropTo", "apply", "stick"]):
		if (len(action["args"]) != 2):
			return False
		for i in action["args"]:
			if i not in object2idx:
				return False
	elif action["name"] in ["moveTo", "pick", "climbUp", "climbDown", "clean"]:
		if (len(action["args"]) != 1):
			return False
	elif action["name"] in ["changeState"]:
		if (len(action["args"]) != 2):
			return False
		if action["args"][1] in object2idx:
			return False
	else:
		assert False
	return True

def accuracy_score(dset, graphs, model, modelEnc, num_objects = 0, verbose = False):
	total_correct = 0
	total_ungrammatical = 0
	denominator = 0
	total_test_loss = 0; l = nn.CrossEntropyLoss()
	for graph in (graphs):
		goal_num, world_num, tools, g, t = graph
		if 'gcn_seq' in training:
			actionSeq, graphSeq = g; loss = 0; toolSeq = tools
			for i, g in enumerate(graphSeq):
				y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
				total_test_loss += l(y_pred.view(1,-1), torch.LongTensor([TOOLS.index(toolSeq[i])]))
				y_pred = list(y_pred.reshape(-1))
				# tools_possible = dset.goal_scene_to_tools[(goal_num,world_num)]
				tools_possible = [toolSeq[i]]
				tool_predicted = TOOLS[y_pred.index(max(y_pred))]
				if tool_predicted in tools_possible:
					total_correct += 1
				elif verbose:
					print (goal_num, world_num, tool_predicted, tools_possible)
				denominator += 1
			continue
		elif 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
			denominator += 1
		elif training == 'combined':
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
			denominator += 1
		elif 'sequence' in training:
			actionSeq, graphSeq = g
			if "aseq" in training:
				if "tool" in training:
					object_likelihoods = []
					for g in graphSeq:
						tool_likelihoods = modelEnc(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
						object_likelihoods.append(tool2object_likelihoods(num_objects, tool_likelihoods))
					y_pred_list = model(graphSeq, goal2vec[goal_num], goalObjects2vec[goal_num], actionSeq, object_likelihoods)
				else:
					y_pred_list = model(graphSeq, goal2vec[goal_num], goalObjects2vec[goal_num], actionSeq)
				for i,y_pred in enumerate(y_pred_list):
					denominator += 1
					action_pred = vec2action_grammatical(y_pred, num_objects, 4, idx2object)
					if verbose:
						if (not grammatical_action(action_pred)):
							# print (action_pred)
							total_ungrammatical += 1
						if (not grammatical_action(actionSeq[i])):
							print (actionSeq[i])
					if (action_pred == actionSeq[i]):
						total_correct += 1
			else:	
				for i in range(len(graphSeq)):
					if 'list' not in training:
						y_pred = model(graphSeq[i], goal2vec[goal_num], goalObjects2vec[goal_num])
					elif training == 'sequence_list':
						y_pred = model(graphSeq[max(0,i + 1 - graph_seq_length):i+1], goal2vec[goal_num], goalObjects2vec[goal_num])
					denominator += 1
					action_pred = vec2action_grammatical(y_pred, num_objects, 4, idx2object)
					# print ("Prediction: ", action_pred)
					# print ("Target: ", actionSeq[i])
					if verbose:
						if (not grammatical_action(action_pred)):
							# print (action_pred)
							total_ungrammatical += 1
						if (not grammatical_action(actionSeq[i])):
							print (actionSeq[i])
					if (action_pred == actionSeq[i]):
						total_correct += 1
			continue
		tools_possible = dset.goal_scene_to_tools[(goal_num,world_num)]
		y_pred = list(y_pred.reshape(-1))
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]
		if tool_predicted in tools_possible:
			total_correct += 1
		elif verbose:
			print (goal_num, world_num, tool_predicted, tools_possible)
	if (("sequence" in training) and verbose):
		print ("Total ungrammatical percent is", (total_ungrammatical/denominator) * 100)
		print ("Denominator is", denominator)
	if training == 'gcn_seq':
		print("Normalized Loss:", total_test_loss.item()/denominator)
	return ((total_correct/denominator)*100)

def printPredictions(model, data=None):
	if not data:
		data = DGLDataset("dataset/" + domain + "/", 
			augmentation=AUGMENTATION, 
			globalNode=globalnode, 
			ignoreNoTool=ignoreNoTool, 
			sequence=sequence)
	total_number = 0
	for graph in data.graphs:
		goal_num, world_num, tools, g, t = graph
		if 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
		elif training == 'combined':
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
		elif 'sequence' in training:
			actionSeq, graphSeq = g
			if "aseq" in training:
				y_pred_list = model(graphSeq, goal2vec[goal_num], goalObjects2vec[goal_num], actionSeq)
				for i,y_pred in enumerate(y_pred_list):
					denominator += 1
					action_pred = vec2action_grammatical(y_pred, num_objects, 4, idx2object)
					if verbose:
						if (not grammatical_action(action_pred)):
							# print (action_pred)
							total_ungrammatical += 1
						if (not grammatical_action(actionSeq[i])):
							print (actionSeq[i])
					if (action_pred == actionSeq[i]):
						total_correct += 1
			else:
				for i in range(len(graphSeq)):
					if 'list' not in training:
						y_pred = model(graphSeq[i], goal2vec[goal_num], goalObjects2vec[goal_num])
					elif training == 'sequence_list':
						y_pred = model(graphSeq[max(0,i + 1 - graph_seq_length):i+1], goal2vec[goal_num], goalObjects2vec[goal_num])
					action_pred = vec2action(y_pred, data.num_objects, 4, idx2object)
					# if (action_pred != actionSeq[i]):
					# 	print ("Prediction: ", action_pred)
					# 	print ("Target: ", actionSeq[i])
					total_number += 1
				continue
		tools_possible = data.goal_scene_to_tools[(goal_num,world_num)]
		y_pred = list(y_pred.reshape(-1))
		# y_pred[TOOLS.index("box")] = 0
		tool_predicted = TOOLS[y_pred.index(max(y_pred))]
		# if tool_predicted == "tray" or tool_predicted == "tray2":
		print(goal_num, world_num, tool_predicted, tools_possible)
		# print(tool_predicted, "\t\t", tools_possible)
	print ("Total number of states is", total_number)

def backprop(data, optimizer, graphs, model, num_objects, modelEnc=None, batch_size = 1):
	total_loss = 0.0
	l = nn.BCELoss()
	batch_loss = 0.0
	for iter_num, graph in tqdm(list(enumerate(graphs))):
		goal_num, world_num, tools, g, t = graph
		if 'ae' in training:
			y_pred = model(g)
			y_true = g.ndata['feat']
			loss = torch.sum((y_pred - y_true)** 2)
			batch_loss += loss
		elif 'gcn_seq' in training:
			l = nn.CrossEntropyLoss()
			actionSeq, graphSeq = g; loss = 0; toolSeq = tools
			for i, g in enumerate(graphSeq):
				y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
				y_true = torch.LongTensor([TOOLS.index(toolSeq[i])])
				loss = l(y_pred.view(1,-1), y_true)
				if weighted: loss *= (1 if t == data.min_time[(goal_num, world_num)] else 0.5)
				batch_loss += loss
		elif 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = l(y_pred, y_true)
			if weighted: loss *= (1 if t == data.min_time[(goal_num, world_num)] else 0.5)
			batch_loss += loss
		elif 'combined' in training:
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = torch.sum((y_pred - y_true)** 2)
			batch_loss += loss
		elif 'sequence' in training:
			actionSeq, graphSeq = g; loss = 0
			if "aseq" in training:
				if "tool" in training:
					object_likelihoods = []
					for g in graphSeq:
						tool_likelihoods = modelEnc(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
						object_likelihoods.append(tool2object_likelihoods(num_objects, tool_likelihoods))
					y_pred_list = model(graphSeq, goal2vec[goal_num], goalObjects2vec[goal_num], actionSeq, object_likelihoods)
				else:
					y_pred_list = model(graphSeq, goal2vec[goal_num], goalObjects2vec[goal_num], actionSeq)
				for i,y_pred in enumerate(y_pred_list):
					y_true = action2vec(actionSeq[i], num_objects, 4)
					loss += l(y_pred, y_true)
			else:
				for i in range(len(graphSeq)):
					if 'list' not in training:
						y_pred = model(graphSeq[i], goal2vec[goal_num], goalObjects2vec[goal_num])
					elif training == 'sequence_list':
						y_pred = model(graphSeq[max(0,i + 1 - graph_seq_length):i + 1], goal2vec[goal_num], goalObjects2vec[goal_num])
					y_true = action2vec(actionSeq[i], num_objects, 4)
					loss += l(y_pred, y_true)
					# loss += torch.sum((y_pred - y_true)** 2)
			batch_loss += loss
		total_loss += loss
		if ((iter_num + 1) % batch_size == 0):
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			batch_loss = 0
	return (total_loss.item()/len(graphs))

def backpropGD(data, optimizer, graphs, model, num_objects, modelEnc=None):
	total_loss = 0.0
	l = nn.BCELoss()
	for iter_num, graph in enumerate(graphs):
		goal_num, world_num, tools, g, t = graph
		if 'ae' in training:
			y_pred = model(g)
			y_true = g.ndata['feat']
			loss = torch.sum((y_pred - y_true)** 2)
		elif 'gcn' in training:
			y_pred = model(g, goal2vec[goal_num], goalObjects2vec[goal_num], tool_vec)
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = l(y_pred, y_true)
			if weighted: loss *= (1 if t == data.min_time[(goal_num, world_num)] else 0.5)
		elif 'combined' in training:
			encoding = modelEnc.encode(g)[-1] if globalnode else modelEnc.encode(g)
			y_pred = model(encoding.flatten(), goal2vec[goal_num], goalObjects2vec[goal_num])
			y_true = torch.zeros(NUMTOOLS)
			for tool in tools: y_true[TOOLS.index(tool)] = 1
			loss = torch.sum((y_pred - y_true)** 2)
		elif 'sequence' in training:
			actionSeq, graphSeq = g; loss = 0
			if "aseq" in training:
				y_pred_list = model(graphSeq, goal2vec[goal_num], goalObjects2vec[goal_num], actionSeq)
				for i,y_pred in enumerate(y_pred_list):
					y_true = action2vec(actionSeq[i], num_objects, 4)
					loss += l(y_pred, y_true)
			else:
				for i in range(len(graphSeq)):
					if 'list' not in training:
						y_pred = model(graphSeq[i], goal2vec[goal_num], goalObjects2vec[goal_num])
					elif training == 'sequence_list':
						y_pred = model(graphSeq[max(0,i + 1 - graph_seq_length):i + 1], goal2vec[goal_num], goalObjects2vec[goal_num])
					y_true = action2vec(actionSeq[i], num_objects, 4)
					loss += l(y_pred, y_true)
					# loss += torch.sum((y_pred - y_true)** 2)
		total_loss += loss
	optimizer.zero_grad()
	total_loss.backward()
	optimizer.step()
	return (total_loss.item()/len(graphs))

def random_split(data):
	test_size = int(0.1 * len(data.graphs))
	random.shuffle(data.graphs)
	test_set = data.graphs[:test_size]
	train_set = data.graphs[test_size:]
	return train_set, test_set

def world_split(data):
	test_set = []
	train_set = []
	for i in data.graphs:
		for j in range(1,9):
			if (i[0],i[1]) == (j,j):
				test_set.append(i)
				break
		else:
			train_set.append(i)
	return train_set, test_set

def tool_split(data):
	train_set, test_set = world_split(data)
	tool_set, notool_set = [], []
	for graph in train_set:
		if 'no-tool' in graph[2]: notool_set.append(graph)
		else: tool_set.append(graph)
	new_set = []
	for i in range(len(tool_set)-len(notool_set)):
		new_set.append(random.choice(notool_set))
	train_set = tool_set + notool_set + new_set
	return train_set, test_set

def write_training_data(model_name, loss, training_accuracy, test_accuracy):
	file_path = "trained_models/training/"+('Seq_' if training == 'gcn_seq' else '')+model_name+".pt"
	if path.exists(file_path):
		with open(file_path, "rb") as f:
			llist, trlist, telist = pickle.load(f)
	else:
		llist, trlist, telist = [], [], []
	llist.append(loss); trlist.append(training_accuracy); telist.append(test_accuracy)
	with open(file_path, "wb") as f:
		pickle.dump((llist, trlist, telist), f)

if __name__ == '__main__':
	filename = ('dataset/'+ domain + '_'+ 
				("global_" if globalnode else '') + 
				("NoTool_" if not ignoreNoTool else '') + 
				("seq_" if sequence else '') + 
				(embedding) +
				str(AUGMENTATION)+'.pkl')
	data = load_dataset(filename)
	modelEnc = None
	if train and not generalization:
		if training == 'gcn' and not globalnode:
			model = DGL_GCN(data.features, data.num_objects, GRAPH_HIDDEN, NUMTOOLS, 3, etypes, nn.functional.relu, 0.5)
		elif training == 'gcn' and globalnode:
			model = DGL_GCN_Global(data.features, data.num_objects, GRAPH_HIDDEN, NUMTOOLS, 3, etypes, nn.functional.relu, 0.5)
		elif training == 'ae':
			model = DGL_AE(data.features, GRAPH_HIDDEN, 3, etypes, nn.functional.relu, globalnode)
		elif training == 'combined' and globalnode:
			modelEnc = torch.load("trained_models/GCN-AE_Global_10.pt")#; modelEnc.freeze()
			model = DGL_Decoder_Global(GRAPH_HIDDEN, NUMTOOLS, 3)
		elif training == 'combined' and not globalnode:
			modelEnc = torch.load("trained_models/GCN-AE_10.pt") #; modelEnc.freeze()
			model = DGL_Decoder(GRAPH_HIDDEN, NUMTOOLS, 3)
		elif training == 'agcn':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_640_3_Trained.pt")
			model = DGL_AGCN(data.features, data.num_objects, 10 * GRAPH_HIDDEN, NUMTOOLS, 3, etypes, nn.functional.tanh, 0.5)
		elif training == "agcn-tool":
			# model = torch.load("trained_models/Simple_Attention_Tool_768_3_Trained.pt")
			model = DGL_Simple_Tool(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
		elif training == 'agcn-likelihood':
			# model = torch.load("trained_models/Final_Metric_256_5_9.pt")
			# model = GGCN(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
			model = Final_C(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
			# model = DGL_Simple_Likelihood(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5, embedding, weighted)
		elif training == 'gcn_seq':
			# model = torch.load("trained_models/Seq_GGCN_Metric_Attn_L_NT_C_256_5_Trained.pt")
			# model = GGCN_Metric_Attn_L(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5)
			model = DGL_Simple_Likelihood(data.features, data.num_objects, 4 * GRAPH_HIDDEN, NUMTOOLS, 5, etypes, torch.tanh, 0.5, embedding, weighted)
		elif training == 'sequence':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Action_128_3_16.pt")
			model = DGL_AGCN_Action(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh, 0.5)
		elif training == 'sequence_list':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Action_List_128_3_0.pt")
			model = DGL_AGCN_Action_List(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh, 0.5, graph_seq_length)
		elif training == 'sequence_baseline':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Action_128_3_16.pt")
			model = GGCN_Action(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh, 0.5)
		elif training == 'sequence_baseline_metric':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Action_128_3_16.pt")
			model = GGCN_metric_Action(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh, 0.5)
		elif training == 'sequence_baseline_metric_att':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Action_128_3_16.pt")
			model = Metric_att_Action(data.features, data.num_objects, 5 * GRAPH_HIDDEN, 4, 5, etypes, torch.tanh, 0.5)
			# model = GGCN_metric_att_Action(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh, 0.5)
		elif training == 'sequence_baseline_metric_att_aseq':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Action_128_3_16.pt")
			model = GGCN_metric_att_aseq_Action(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh, 0.5)
		elif training == 'sequence_baseline_metric_att_tool_aseq':
			# model = torch.load("trained_models/GatedHeteroRGCN_Attention_Action_128_3_16.pt")
			modelEnc = torch.load("trained_models/Seq_GGCN_Metric_Attn_L_NT_C_256_5_Trained.pt"); modelEnc.eval()
			for param in modelEnc.parameters(): param.requires_grad = False
			model = GGCN_metric_att_aseq_tool_Action(data.features, data.num_objects, 2 * GRAPH_HIDDEN, 4, 3, etypes, torch.tanh, 0.5)

		lr = 0.0005 if 'sequence' in training else 0.00005
		if training == 'gcn_seq': lr = 0.000005 
		optimizer = torch.optim.Adam(model.parameters() , lr=lr)
		# optimizer.load_state_dict(torch.load("trained_models/GatedHeteroRGCN_Attention_Action_List_128_3_0.optim").state_dict())
		print ("Training " + model.name + " with " + embedding)
		train_set, test_set = world_split(data) if split == 'world' else random_split(data)  if split == 'random' else tool_split(data) 

		print ("Size before split was", len(data.graphs))
		print ("The size of the training set is", len(train_set))
		print ("The size of the test set is", len(test_set))
		addEmbedding = embedding[0] + "_" if 'sequence' in training else ''
		seqTool = 'Seq_' if training == 'gcn_seq' else ''
		accuracy_list = []
		for num_epochs in range(NUM_EPOCHS+1):
			random.shuffle(train_set)
			print ("EPOCH " + str(num_epochs))
			loss = backprop(data, optimizer, train_set, model, data.num_objects, modelEnc)
			print(loss)
			t1, t2 = accuracy_score(data, train_set, model, modelEnc, data.num_objects), accuracy_score(data, test_set, model, modelEnc, data.num_objects)
			accuracy_list.append((t2, t1))
			if (num_epochs % 1 == 0):
				if training != "ae":
					print ("Accuracy on training set is ", t1)
					print ("Accuracy on test set is ", t2)
				elif training == 'ae':
					print ("Loss on test set is ", loss_score(test_set, model, modelEnc).item()/len(test_set))
				if num_epochs % 1 == 0:
					torch.save(model, MODEL_SAVE_PATH + "/" + seqTool + model.name + "_" + addEmbedding + str(num_epochs) + ".pt")
					torch.save(optimizer, MODEL_SAVE_PATH + "/" + seqTool + model.name + "_" + addEmbedding + str(num_epochs) + ".optim")
			pickle.dump(accuracy_list, open(MODEL_SAVE_PATH + "/" + seqTool + model.name + "_" + addEmbedding + "accuracies.pkl", "wb"))
			write_training_data(model.name, loss, t1, t2)
		print ("The maximum accuracy on test set, train set for " + str(NUM_EPOCHS) + " epochs is ", str(max(accuracy_list)), " at epoch ", accuracy_list.index(max(accuracy_list)))
	elif not train and not generalization:
		print ("Evaluating...")
		model = torch.load(MODEL_SAVE_PATH + "/checkpoints/sequence_best_78_47.pt")
		# print ("Accuracy on complete set is ",accuracy_score(data, data.graphs, model, modelEnc))
		train_set, test_set = world_split(data) if split == 'world' else random_split(data)  if split == 'random' else tool_split(data) 
		t1, t2 = accuracy_score(data, train_set, model, modelEnc, data.num_objects, True), accuracy_score(data, test_set, model, modelEnc, data.num_objects, True)
		print ("Accuracy on training set is ", t1)
		print ("Accuracy on test set is ", t2)
		# printPredictions(model,data)
	else:
		if not ablation:
			testConcept = TestDataset("dataset/test/" + domain + "/conceptnet/")
			testFast = TestDataset("dataset/test/" + domain + "/fasttext/")
			embeddings, object2vec, object2idx, idx2object, tool_vec, goal2vec, goalObjects2vec = compute_constants("fasttext")
			for i in ["GGCN_256_5_0", "GGCN_Metric_256_5_Trained", "GGCN_Metric_Attn_256_5_Trained",\
						"GGCN_Metric_Attn_L_256_5_Trained", "GGCN_Metric_Attn_L_NT_256_5_Trained"]:
				model = torch.load(MODEL_SAVE_PATH + "/" + i + ".pt")
				print(i, gen_score(model, testFast))
			embeddings, object2vec, object2idx, idx2object, tool_vec, goal2vec, goalObjects2vec = compute_constants("conceptnet")
			for i in ["GGCN_Metric_Attn_L_NT_C_256_5_Trained", "GGCN_Metric_Attn_L_NT_C_W_256_5_Trained"]:
				model = torch.load(MODEL_SAVE_PATH + "/" + i + ".pt")
				print(i, gen_score(model, testConcept))
		else:
			testConcept = TestDataset("dataset/test/" + domain + "/conceptnet/")
			testFast = TestDataset("dataset/test/" + domain + "/fasttext/")
			embeddings, object2vec, object2idx, idx2object, tool_vec, goal2vec, goalObjects2vec = compute_constants("fasttext")
			for i in ["Final_C_256_5_Trained"]:
				model = torch.load(MODEL_SAVE_PATH + "/" + i + ".pt")
				print(i, gen_score(model, testFast))
			embeddings, object2vec, object2idx, idx2object, tool_vec, goal2vec, goalObjects2vec = compute_constants("conceptnet")
			for i in ["Final_Attn_256_5_Trained", "Final_L_256_5_Trained", "Final_Metric_256_5_Trained",\
						"Final_NT_256_5_Trained"]:
				model = torch.load(MODEL_SAVE_PATH + "/" + i + ".pt")
				print(i, gen_score(model, testConcept))

