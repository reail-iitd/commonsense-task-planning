from src.datapoint import *
import pickle
import json
from extract_vectors import load_all_vectors
from copy import deepcopy
from src.GNN.CONSTANTS import TOOLS2, domain
from os import listdir

conceptnet = load_all_vectors("jsons/embeddings/conceptnet.txt") # {} #
fasttext = load_all_vectors("jsons/embeddings/fasttext.txt") # {} #
with open('jsons/embeddings/conceptnet.vectors') as handle: ce = json.load(handle)
with open('jsons/embeddings/fasttext.vectors') as handle: fe = json.load(handle)

directory_home = {1: "dataset/home/goal2-fruits-cupboard/world_home0/",
			 2: "dataset/home/goal1-milk-fridge/world_home1/",
			 3: "dataset/home/goal3-clean-dirt/",
			 4: "dataset/home/goal6-bottles-dumpster/",
			 5: "dataset/home/goal4-stick-paper/",
			 6: "dataset/home/goal2-fruits-cupboard/",
			 7: "dataset/home/goal8-light-off/",
			 8: "dataset/home/goal6-bottles-dumpster/",
			 9: "dataset/home/goal1-milk-fridge/world_home0/"}

goal_num_home = {1:2 ,2:1 ,3:3 ,4:6 ,5:4 ,6:2 ,7:8, 8:6, 9:1}

tools_home = {1: ["tray2", "stick"],
		 2: ["no-tool"],
		 3: ["stool", "vacuum", "sponge"],
		 4: ["box", "stool", "no-tool"],
		 5: ["stool", "tape", "stick"],
		 6: ["stool", "stick", "book", "box", "no-tool"],
		 7: ["stick", "no-tool"],
		 8: ["stick", "stool", "no-tool", "tray", "tray2", "chair"],
		 9: ["stool", "tray", "stick"]}

def writeFiles(number, path, d):
	i = len(listdir(path))
	for j in range(number):
		f = open(path + str(i) + ".graph", "w+") 
		f.write(json.dumps(d, indent=2))
		f.close()
		i += 1

def formTestData(testnum):
	all_files = os.walk(directory_home[testnum])
	for path, dirs, files in all_files:
		if (len(files) > 0):
			for file in files:
				file_path = path + "/" + file
				with open(file_path, 'rb') as f:
					datapoint = pickle.load(f)
				for e in [(conceptnet, "conceptnet", ce), (fasttext, "fasttext", fe)]:	
					d = {"goal_num": goal_num_home[testnum], "tools": tools_home[testnum]} 
					enew = deepcopy(e[2])
					if testnum == 3: enew["mop"] = [0] * 300
					elif testnum == 4: enew["box"] = e[0]["crate"] 
					elif testnum == 5: enew["glue"] = [0] * 300
					elif testnum == 6: enew["apple"] = e[0]["guava"] 
					elif testnum == 7: enew["stool"] = e[0]["headphone"] 
					elif testnum == 8: enew["box"] = [0] * 300
					elif testnum == 9: enew["stool"] = e[0]["seat"] 
					g = datapoint.getGraph(embeddings = enew)
					d["graph_0"] = g["graph_0"]
					d["tool_embeddings"] = [enew[i] for i in TOOLS2]
					writeFiles(5 if testnum <= 2 or testnum == 9 else 1, "dataset/test/home/" + e[1] + "/test"+ str(testnum) + "/", d)

directory_factory = {1: "dataset/factory/goal7-clean-water/",
			 2: "dataset/factory/goal3-board-wall/",
			 3: "dataset/factory/goal1-crates-platform/",
			 4: "dataset/factory/goal8-clean-oil/",
			 5: "dataset/factory/goal2-paper-wall/",
			 6: "dataset/factory/goal5-assemble-parts/",
			 7: "dataset/factory/goal4-generator-on/",
			 8: "dataset/factory/goal6-tools-workbench/world_factory5/"}

goal_num_factory = {1:7 ,2:3 ,3:1 ,4:8 ,5:2 ,6:5 ,7:4, 8:6}

tools_factory = {1: ["mop"],
		 2: ["drill", "3d_printer", "screwdriver", "hammer", "stool", "stick", "lift", "ladder"],
		 3: ["no-tool", "trolley", "stool", "stick", "ladder"],
		 4: ["mop"],
		 5: ['stool', 'glue', 'ladder', 'tape', 'stick', 'lift'],
		 6: ['welder', 'spraypaint', 'toolbox', 'lift'],
		 7: ['no-tool', 'stick', 'lift', 'ladder'],
		 8: ['toolbox', '3d_printer']}


def formTestDataFactory(testnum):
	all_files = os.walk(directory_factory[testnum])
	for path, dirs, files in all_files:
		if (len(files) > 0):
			for file in files:
				file_path = path + "/" + file
				with open(file_path, 'rb') as f:
					datapoint = pickle.load(f)
				for e in [(conceptnet, "conceptnet", ce), (fasttext, "fasttext", fe)]:	
					d = {"goal_num": goal_num_factory[testnum], "tools": tools_factory[testnum]} 
					enew = deepcopy(e[2])
					if testnum == 1: enew["blow_dryer"] = [0] * 300
					elif testnum == 2: enew["brick"] = [0] * 300
					elif testnum == 3: enew["lift"] = e[0]["headphone"]
					elif testnum == 4: enew["mop"] = e[0]["mop"] if 'c' in e[1] else e[0]['washcloth']
					elif testnum == 5: enew["glue"] = [0] * 300
					elif testnum == 6: enew["toolbox"] = e[0]["box"]
					elif testnum == 7: enew["wood_cutter"] = e[0]["table"]
					g = datapoint.getGraph(embeddings = enew)
					d["graph_0"] = g["graph_0"]
					if testnum == 8:
						for i in d["graph_0"]["nodes"]: 
							if i["name"] == "screwdriver" and 'To_Print' in i['states']: i["states"].remove('To_Print'); i['states'].append('Printed')
					d["tool_embeddings"] = [enew[i] for i in TOOLS2]
					writeFiles(5 if testnum == 8 else 1, "dataset/test/factory/" + e[1] + "/test"+ str(testnum) + "/", d)

