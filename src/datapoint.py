from copy import deepcopy
from src.utils import *
import json
from tqdm import tqdm
from random import randint

tools = ['stool', 'tray', 'tray2', 'lift', 'big-tray', 'book', 'box', 'chair',\
		'stick', 'glue', 'tape', 'mop', 'sponge', 'vacuum', 'drill', 'screwdriver',\
		'hammer', 'ladder', 'trolley', 'brick', 'blow_dryer', 'spraypaint', 'welder',\
		'toolbox', 'wood_cutter', '3d_printer']

printable = ['screw', 'nail', 'screwdriver', 'hammer']

skip = ['ur5', 'cupboard_back', 'fridge_back', 'ramp_tape', 'lift_base', 'ledge']

objects = None
with open('jsons/objects.json', 'r') as handle:
    objects = json.load(handle)['objects']

allStates = None
with open('jsons/states.json', 'r') as handle:
    allStates = json.load(handle)

class Datapoint:
	def __init__(self):
		# World
		self.world = ""
		# Goal
		self.goal = ""
		# Robot position list
		self.position = []
		# Metrics of all objects
		self.metrics = []
		# Sticky objects
		self.sticky = []
		# Fixed objects
		self.fixed = []
		# Has cleaner
		self.cleaner = []
		# Action
		self.actions = []
		# Constraints
		self.constraints = []
		# Symbolic actions
		self.symbolicActions = []
		# Objects on
		self.on = []
		# Objects Cleaned
		self.clean = []
		# Stick with object
		self.stick = []
		# Objects fueled
		self.fueled = []
		# Objects welded
		self.welded = []
		# Objects painted
		self.painted = []
		# Objects drilled
		self.drilled = []
		# Objects cut
		self.cut = []
		# Time
		self.time = 0

	def addPoint(self, pos, sticky, fixed, cleaner, action, cons, metric, on, clean, stick, welded, drilled, painted, fueled, cut):
		self.position.append(deepcopy(pos))
		self.sticky.append(deepcopy(sticky))
		self.fixed.append(deepcopy(fixed))
		self.cleaner.append(deepcopy(cleaner))
		self.actions.append(deepcopy(action))
		self.constraints.append(deepcopy(cons))
		self.metrics.append(deepcopy(metric))
		self.on.append(deepcopy(on))
		self.clean.append(deepcopy(clean))
		self.stick.append(deepcopy(stick))
		self.welded.append(deepcopy(welded))
		self.drilled.append(deepcopy(drilled))
		self.painted.append(deepcopy(painted))
		self.fueled.append(deepcopy(fueled))
		self.cut.append(deepcopy(cut))

	def addSymbolicAction(self, HLaction):
		self.symbolicActions.append(HLaction)

	def toString(self, delimiter='\n', subSymbolic=False, metrics=False):
		string = "World = " + self.world + "\nGoal = " + self.goal
		string += '\nSymbolic actions:\n'
		for action in self.symbolicActions:
			if str(action[0]) == 'E' or str(action[0]) == 'U':
				string = string + action + '\n'
				continue
			string = string + "\n".join(map(str, action)) + '\n'
		if not subSymbolic:
			return string
		string += 'States:\n'
		for i in range(len(self.position)):
			string = string + 'State ' + str(i) + ' ----------- ' + delimiter + \
				'Robot position - ' + str(self.position[i]) + delimiter + \
				'Sticky - ' + str(self.sticky[i]) + delimiter + \
				'Fixed - ' + str(self.fixed[i]) + delimiter + \
				'Cleaner? - ' + str(self.cleaner[i]) + delimiter + \
				'Objects-Cleaned? - ' + str(self.clean[i]) + delimiter + \
				'Stick with robot? - ' + str(self.stick[i]) + delimiter + \
				'Objects On - ' + str(self.on[i]) + delimiter + \
				'Objects welded - ' + str(self.welded[i]) + delimiter + \
				'Objects drilled - ' + str(self.drilled[i]) + delimiter + \
				'Objects painted - ' + str(self.painted[i]) + delimiter + \
				'Objects fueled - ' + str(self.fueled[i]) + delimiter + \
				'Objects cut - ' + str(self.cut[i]) + delimiter + \
				'Action - ' + str(self.actions[i]) + delimiter + \
				'Constraints - ' + str(self.constraints[i]) + delimiter
			if metrics:
				string = string + 'All metric - ' + str(self.metrics) + delimiter
		return string

	def readableSymbolicActions(self):
		string = 'Symbolic actions:\n\n'
		for action in self.symbolicActions:
			if str(action[0]) == 'E' or str(action[0]) == 'U':
				string = string + action + '\n'
				continue
			assert len(action) == 1
			dic = action[0]
			l = dic["args"]
			string = string + dic["name"] + "(" + str(l[0])
			for i in range(1, len(l)):
				string = string + ", " + str(l[i])
			string = string + ")\n"
		return string

	def getGraph(self, index=0, distance=False, sceneobjects=[], embeddings={}):
		world = 'home' if 'home' in self.world else 'factory' if 'factory' in self.world else 'outdoor'
		metrics = self.metrics[index]
		sceneobjects = list(metrics.keys()) if len(sceneobjects) == 0 else sceneobjects
		if 'factory' in self.world:
			for ob in printable:
				if not ob in sceneobjects: sceneobjects.append(ob)
		globalidlookup = globalIDLookup(sceneobjects, objects)
		nodes = []
		for obj in sceneobjects:
			if obj in skip: continue
			node = {}; objID = globalidlookup[obj]
			node['id'] = objID
			node['name'] = obj
			node['properties'] = objects[objID]['properties']
			if 'Movable' in node['properties'] and obj in self.fixed[index]: node['properties'].remove('Movable')
			states = []
			if obj in 'dumpster': states.append('Outside')
			else: states.append('Inside')
			if 'Switchable' in node['properties']:
				states.append('On') if obj in self.on[index] else states.append('Off')
			if 'Can_Open' in node['properties']:
				states.append('Close') if isInState(obj, allStates[world][obj]['close'], metrics[obj]) else states.append('Open')
			if 'Can_Lift' in node['properties']:
				states.append('Up') if isInState(obj, allStates[world][obj]['up'], metrics[obj]) else states.append('Down')
			if 'Stickable' in node['properties']:
				states.append('Sticky') if obj in self.sticky[index] else states.append('Non_Sticky')
			if 'Is_Dirty' in node['properties']:
				states.append('Dirty') if not obj in self.clean[index] else states.append('Clean')
			if 'Movable' in node['properties']:
				states.append('Grabbed') if grabbedObj(obj, self.constraints[index]) else states.append('Free')
			if 'Weldable' in node['properties']:
				states.append('Welded') if obj in self.welded[index] else states.append('Not_Welded')
			if 'Drillable' in node['properties']:
				states.append('Drilled') if obj in self.drilled[index] else states.append('Not_Drilled')
			if 'Drivable' in node['properties']:
				states.append('Driven') if obj in self.fixed[index] else states.append('Not_Driven')
			if 'Can_Fuel' in node['properties']:
				states.append('Fueled') if obj in self.fueled[index] else states.append('Not_Fueled')
			if 'Cuttable' in node['properties']:
				states.append('Cut') if obj in self.cut[index] else states.append('Not_Cut')
			if 'Can_Paint' in node['properties']:
				states.append('Painted') if obj in self.cut[index] else states.append('Not_Painted')
			if 'Printable' in node['properties']:
				states.append('To_Print') if obj in metrics.keys() else states.append('Printed')
			try: states.append('Different_Height') if abs(metrics[obj][0][2]-metrics["husky"][0][2]) > 1 else states.append("Same_Height")
			except: states.append('Different_Height') if abs(metrics['3d_printer'][0][2]-metrics["husky"][0][2]) > 1 else states.append("Same_Height")
			node['states'] = states
			try: node['position'] = metrics[obj]
			except: node['position'] = metrics['3d_printer']
			node['size'] = objects[objID]['size']
			node['vector'] = embeddings[obj]
			nodes.append(node)
		edges = []
		for i in range(len(sceneobjects)):
			obj1 = sceneobjects[i]
			if obj1 in skip or not obj1 in metrics.keys(): continue
			for j in range(len(sceneobjects)):
				obj2 = sceneobjects[j]
				if obj2 in skip or i == j or not obj2 in metrics.keys(): continue
				obj1ID = globalidlookup[obj1]; obj2ID = globalidlookup[obj2]
				if checkNear(obj1, obj2, metrics):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'Close'}) 
				if checkIn(obj1, obj2, objects[obj1ID], objects[obj2ID], metrics, self.constraints[index]):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'Inside'}) 
				if checkOn(obj1, obj2, objects[obj1ID], objects[obj2ID], metrics, self.constraints[index]):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'On'}) 
				if obj2 == 'walls' and 'Stickable' in objects[obj1ID]['properties'] and isInState(obj1, allStates[world][obj1]['stuck'], metrics[obj1]):
					edges.append({'from': obj1ID, 'to': obj2ID, 'relation': 'Stuck'}) 
				if distance:
					edges.append({'from': obj1ID, 'to': obj2ID, 'distance': getDirectedDist(obj1, obj2, metrics)})
		return {'graph_'+str(index): {'nodes': nodes, 'edges': edges}}

	def getAugmentedGraph(self, index=0, distance=False, remove=5):
		allObjects = list(self.metrics[index].keys())
		actionObjects = []
		for action in self.actions:
			if str(action[0]) == 'E' or str(action[0]) == 'U': continue
			for i in range(1, len(action)):
				if action[i] in allObjects and not action[i] in actionObjects and 'str' in str(type(action[i])):
					actionObjects.append(action[i])
		actionObjects.append('husky')
		for j in range(randint(1, remove)):
			obj = allObjects[randint(0, len(allObjects)-1)]
			if obj in allObjects and not obj in actionObjects:
				allObjects.remove(obj)
		return self.getGraph(index, distance, sceneobjects=allObjects)

	def getTools(self, returnNoTool=False):
		goal_objects = getGoalObjects(self.world, self.goal)
		usedTools = []
		for action in self.actions:
			if 'Start' in action or 'Error' in action: continue
			for obj in action[1:]:
				if (not obj in goal_objects) and (not obj in usedTools) and obj in tools:
					usedTools.append(obj)
		if returnNoTool and len(usedTools) == 0: usedTools.append("no-tool")
		return usedTools

	def totalTime(self):
		time = 0
		for i in range(len(self.actions)):
			action = self.actions[i][0]
			if action == 'S':
				continue
			dt = 0
			if action == 'moveTo' or action == 'moveToXY' or action == 'moveZ':
				x1 = self.position[i][0]; y1 = self.position[i][1]; o1 = self.position[i][3]
				if 'list' in str(type(self.actions[i][1])):
					x2 = self.actions[i][1][0]; y2 = self.actions[i][1][1]
				else:
					x2 = self.metrics[i-1][self.actions[i][1]][0][0]; y2 = self.metrics[i-1][self.actions[i][1]][0][1]
				robot, dest = o1%(2*math.pi), math.atan2((y2-y1),(x2-x1))%(2*math.pi)
				left = (robot - dest)%(2*math.pi); right = (dest - robot)%(2*math.pi)
				dt = 100000 * abs(min(left, right)) # time for rotate
				dt += 2000 * abs(max(1.2, distance.euclidean((x1, y1, 0), (x2, y2, 0))) - 1.2) # time for move
			elif action == 'move':
				x1 = self.position[i][0]; y1 = self.position[i][1]; o1 = self.position[i][3]
				x2 = -2; y2 = 3
				dt = 100000 * abs(math.atan2(y2-y1,x2-x1) % (2*math.pi) - (o1%(2*math.pi)))
				dt += 2000 * abs(max(0.2, distance.euclidean((x1, y1, 0), (x2, y2, 0))) - 0.2)
			elif action == 'constrain' or action == 'removeConstraint' or action == 'changeWing':
				dt = 100
			elif action == 'climbUp' or action == 'climbDown' or action == 'changeState':
				dt = 120
			time += dt
		return time
