import io
import numpy as np
from tqdm import tqdm
import pickle
import json

def get_objects():
	objlist = []
	with open('jsons/objects.json', 'r') as handle:
	    objects = json.load(handle)['objects']
	    for obj in objects:
	   		objlist.append(obj['name'])
	return objlist

def load_all_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def load_vectors(fname, objs):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in objs:
        	data[tokens[0]] = list(map(float, tokens[1:]))
    for i in objs:
    	if not i in data.keys():
    		data[i] = []
    		print(i)
    return data

def form_goal_vec(data, text):
    goal_vec = np.zeros(300)
    for j in text.split():
        goal_vec += data[j]
    goal_vec /= len(text.split())
    return goal_vec

def form_file(filename):
	data = load_vectors(filename, get_objects())
	with open('jsons/embeddings/'+filename.split('.')[0]+'.vectors', "w") as fp:
		fp.write("{\n")
		for i in data:
			fp.write('\t\t"' + i + '": ' + str(data[i]) + ',\n')
		fp.write("}")

