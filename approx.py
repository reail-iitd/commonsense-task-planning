import os
import time
import pdb
import pybullet as p
import time
import os, shutil
from src.initialise import *
from src.parser import *
from src.ur5 import *
from src.utils import *
from src.basic_actions import *
from src.actions import *
from src.datapoint import Datapoint
from src.GNN.dataset_utils import *
from operator import sub
import math
import pickle
from operator import add

class Args():
  goal = ''
  world = ''
  speed = 1

GOAL_LISTS = \
{'home': ["goal1-milk-fridge.json", "goal2-fruits-cupboard.json", "goal3-clean-dirt.json", "goal4-stick-paper.json", "goal5-cubes-box.json", "goal6-bottles-dumpster.json", "goal7-weight-paper.json", "goal8-light-off.json"],
'factory': ["goal1-crates-platform.json", "goal2-paper-wall.json", "goal3-board-wall.json", "goal4-generator-on.json", "goal5-assemble-parts.json", "goal6-tools-workbench.json", "goal7-clean-water.json", "goal8-clean-oil.json"]}

object_file = "jsons/objects.json"
wings_file = "jsons/wings.json"
tolerance_file = "jsons/tolerance.json"
goal_file = "jsons/goal.json"

#Number of steps before image capture
COUNTER_MOD = 50

# Enclosures
enclosures = ['fridge', 'cupboard']

# Semantic objects
# Sticky objects
sticky = []
# Fixed objects
fixed = []
# Objects on
on = ['light']
# Has been fueled
fueled = []
# Cut objects
cut = []
# Has cleaner
cleaner = False
# Has stick
stick = False
# Objects cleaned
clean = []
# Objects drilled
drilled = []
# Objects welded
welded = []
# Objects painted
painted = []
# Datapoint
datapoint = Datapoint()
# Metrics
metrics = {}
# Robot position
x1, y1, z1, o1 = 0, 0, 0, 0

def load_world(world_file, object_file):
  metrics, tolerances, properties, ur5_dist, states = {}, {}, {}, {}, {}
  cons_cpos_lookup, cons_pos_lookup = {}, {}
  with open(world_file, 'r') as handle:
    world = json.load(handle)
  with open(object_file, 'r') as handle:
    all_objects = json.load(handle)['objects']
  for obj in world['entities']:
    for o in all_objects:
      if o['name'] == obj['name']: break
    metrics[obj['name']] = [obj['position'], obj['orientation']]
    states[obj['name']] = obj['states'] if len(obj['states']) > 0 else {}
    tolerances[obj['name']] = o['tolerance']
    properties[obj['name']] = o['properties']
    ur5_dist[obj['name']] = o['ur5_dist'] if obj['name'] not in ['stool', 'box'] else [0.3, 0, 0.7]
    cons_cpos_lookup[obj['name']] = o['constraint_cpos']
    cons_pos_lookup[obj['name']] = o['constraint_pos']
  return metrics, tolerances, properties, cons_cpos_lookup, cons_pos_lookup, ur5_dist, states

def start(input_args):
  # Initialize husky and ur5 model
  global husky,robotID, object_lookup, id_lookup, horizontal_list, ground_list,fixed_orientation,tolerances, properties,cons_cpos_lookup,cons_pos_lookup, cons_link_lookup,ur5_dist,states,wings,gotoWing,constraints,constraint,x1, y1, o1
  global x1, y1, z1, o1, light, args, speed, sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted, datapoint, metrics

  sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted, datapoint = [], [], ['light'], [], [], False, False, [], [], [], [], Datapoint()
  # Add input arguments
  args = input_args
  speed = args.speed

  ( metrics,
    tolerances, 
    properties,
    cons_cpos_lookup,
    cons_pos_lookup, 
    ur5_dist,
    states) = load_world(args.world, object_file)

  # Position of the robot
  x1, y1, z1, o1 = 0, 0, 0, 0
  constraint = 0

  # Dict of constraints with target: [obj1, obj2, ...]
  constraints = {'husky': ['ur5'], "ur5": []}
  try:
      g = args.goal.split("/")[-1].split(".")[0]; w = args.world.split('/')[-1].split(".")[0]
  except:
      g = args.goal.split("/")[-1].split(".")[0]; w = args.world.split('/')[-1].split(".")[0]
  datapoint.world = w
  datapoint.goal = g
 
def undo():
  global world_states, x1, y1, o1, imageCount, constraints, on, datapoint
  datapoint.addSymbolicAction("Undo")
  datapoint.addPoint(None, None, None, None, 'Undo', None, None, None, None, None, None, None, None, None, None)
  x1, y1, o1, constraints, world_states = restoreOnInput(world_states, x1, y1, o1, constraints)
  _, imageCount = saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, on)

keyboard = False

def getGoalConstraints():
  gcons = {}
  for target in constraints.keys():
    for obj in constraints[target]:
      gcons[obj] = (target, 0)
  return gcons

def fct(o):
  # proxy for find constrained to
  for t in constraints.keys():
    if o in constraints[t]: return t
  return ""

def fcw(o):
    return constraints[o]

def closed(o):
  global metrics
  positionAndOrientation = states[o]["close"]
  q = p.getQuaternionFromEuler(positionAndOrientation[1])
  if metrics[o][1] == []:
    metrics[o][1] = [0,0,0]
  q1 = p.getQuaternionFromEuler(metrics[o][1])
  ((x1, y1, z1), (a1, b1, c1, d1)) = (metrics[o][0], q1)
  ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
  closed = (abs(x2-x1) <= 0.01 and 
          abs(y2-y1) <= 0.01 and 
          abs(a2-a1) <= 0.01 and 
          abs(b2-b2) <= 0.01 and 
          abs(c2-c1) <= 0.01 and 
          abs(d2-d2) <= 0.01)
  return closed

def cin(o):
  for enclosure in enclosures:
      if not enclosure in metrics.keys(): continue
      if closed(enclosure):
          (x1, y1, z1) = metrics[o][0]
          (x2, y2, z2) = metrics[enclosure][0]
          (l, w, h) = 1.0027969752543706, 0.5047863562602029, 1.5023976731489332
          inside = abs(x2-x1) < 0.5*l and abs(y2-y1) < 1.4*w and abs(z1-z2) < 0.6*h
          # print(enclosure, inside)
          # print((x1, y1, z1), (x2, y2, z2), abs(x2-x1), abs(y2-y1), abs(z2-z1), 0.5*l, 1.5*w,0.6*h)
          tgt = fct(o)
          while not (tgt == "" or tgt == enclosure):
              tgt = fct(tgt)        
          if inside or (tgt == enclosure): return True
  return False

def listSum(l1, l2):
  return list( map(add, l1, l2) )

def updateMetrics():
  global metrics, o1, x1, y1, z1
  metrics['ur5'][0] = [x1, y1, z1]
  metrics['husky'][0] = [x1, y1, z1]
  for convergence in range(10):
    for target in constraints.keys():
      if target == 'ur5':
        for obj in constraints[target]:
          metrics[obj][0][0] = metrics[target][0][0] +  ur5_dist[obj][0] * math.cos(o1)
          metrics[obj][0][1] = metrics[target][0][1] +  ur5_dist[obj][0] * math.sin(o1)
      else:
        for i, obj in enumerate(constraints[target]):
          metrics[obj][0] = listSum(metrics[target][0], cons_pos_lookup[target][i])
  return metrics

def instate(o, st):
  global metrics
  # print(st)
  positionAndOrientation = states[o][st]
  q = p.getQuaternionFromEuler(positionAndOrientation[1])
  if metrics[o][1] == []:
    metrics[o][1] = [0,0,0]
  q1 = p.getQuaternionFromEuler(metrics[o][1])
  ((x1, y1, z1), (a1, b1, c1, d1)) = (metrics[o][0], q1)
  ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
  closed = (abs(x2-x1) <= 0.01 and 
          abs(y2-y1) <= 0.01 and 
          abs(a2-a1) <= 0.01 and 
          abs(b2-b2) <= 0.01 and 
          abs(c2-c1) <= 0.01 and 
          abs(d2-d2) <= 0.01)
  return closed

def cg(goal_file, constraints, states, on, clean, sticky, fixed, drilled, welded, painted):
    if not goal_file:
        return False
    with open(goal_file, 'r') as handle:
        file = json.load(handle)
    goals = file['goals']
    # print(constraints, on)
    success = True
    for goal in goals:
        obj = goal['object']
        if obj == 'light':
            if obj in on:
                success = False

        if obj == 'generator':
            if not obj in on:
                success = False

        if 'part' in obj:
            success = success and obj in welded and obj in painted

        if 'paper' in obj and goal['state'] == "":
            tgt = fcw(obj)
            # print(tgt)
            heavy = False
            for t in tgt:
                if not (t == "" or 'paper' in t):
                    heavy = True
            success = success and heavy

        if obj == 'dirt' or obj == "water" or obj == "oil":
            success = success and obj in clean

        if goal['target'] != "":
            tgt = fct(obj)
            while not (tgt == "" or tgt == goal['target']):
                tgt = fct(tgt)
            success = success and (tgt == goal['target'])

        if goal['state'] != "":
            finalstate = goal['state']
            if finalstate == 'stuck' and not obj in sticky:
                success = False
            if finalstate == 'fixed':
                finalstate = 'stuck'
                success = (success and (
                            ('nail' in fcw(obj) 
                                and 'nail' in fixed) or
                            ('screw' in fcw(obj) 
                                and 'screw' in fixed)))
            done = instate(obj, finalstate)
            success = success and done

        if goal['position'] != "":
            pos = metrics[obj][0]
            goal_pos = metrics[goal['position']][0]
            if abs(distance.euclidean(pos, goal_pos)) > abs(goal['tolerance']):
                success = False
    return success

def executeHelper(actions, goal_file=None, queue_for_execute_to_stop = None, saveImg = True):
  global x1, y1, o1, z1, world_states, dist, yaw, pitch, camX, camY, imageCount, cleaner, on, datapoint, clean, stick, keyboard, drilled, welded, painted, fueled, cut
  global light, args, speed, sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted, datapoint, metrics, constraints
  # List of low level actions
  datapoint.addSymbolicAction(actions['actions'])
  actions = convertActions(actions, args.world)
  action_index = 0
  done = False; done1 = False
  waiting = False
  datapoint.addPoint([x1, y1, z1, o1], sticky, fixed, cleaner, 'Start', getGoalConstraints(), metrics, on, clean, stick, welded, drilled, painted, fueled, cut)

  while(True):
      if action_index >= len(actions):
        return cg(goal_file, constraints, states, on, clean, sticky, fixed, drilled, welded, painted)

      inpAction = actions[action_index][0]
      # print("\n---", actions[action_index])
      # print("Robot: ", [x1, y1, z1])
      # print("milk: ", metrics['milk'][0])
      # print(constraints)
      # print("o1: ", o1*360/(2*math.pi))

      if(inpAction == "move"):
        if "husky" in fixed:
          raise Exception("Husky can not move as it is on a stool")    
        target = actions[action_index][1]
        x2, y2, z2 = target[0], target[1], target[2]
        x1, y1, z1, o1, done = x2, y2, z1, math.atan2((y2-y1),(x2-x1))%(2*math.pi), True

      elif(inpAction == "moveZ"):
        if "husky" in fixed:
              raise Exception("Husky can not move as it is on a stool") 
        if (actions[action_index][1][0] == -1.5
            and actions[action_index][1][1] == 1.5
            and actions[action_index][1][2] == 1):
            if (fct('ramp') != 'floor_warehouse'):
              raise Exception("Can not move up without ramp")
        target = metrics[actions[action_index][1]][0]
        x2, y2, z2 = target[0], target[1], target[2]
        x1, y1, z1, o1, done = x2, y2, z2, math.atan2((y2-y1),(x2-x1))%(2*math.pi), True

      elif(inpAction == "moveTo"):
        t = actions[action_index][1]
        target = np.array(metrics[t][0]); cur = np.array([x1, y1, z1])
        vec = target - cur
        objDistance = np.linalg.norm(vec)
        # print(cur)
        # print(target)
        # print(objDistance)
        if objDistance > 10.3 and "husky" in fixed:
              raise Exception("Husky can not move as it is on a stool")  
        if abs(metrics[t][0][2] - z1) > 1.8 and not stick:
              raise Exception("Object on different height, please use stool")
        if metrics[t][0][2] - z1 < -1.1:
              raise Exception("Object on lower level, please move down")
        cur = cur + (objDistance - tolerances[t]/2) * (vec / objDistance)
        # print(cur)
        x2, y2, z2 = target[0], target[1], target[2]
        x1, y1, z1, o1, done = cur[0], cur[1], z1, math.atan2((y2-y1),(x2-x1))%(2*math.pi), True

      elif(inpAction == "moveToXY"):
        t = actions[action_index][1]
        target = np.array(metrics[t][0]); cur = np.array([x1, y1, z1])
        vec = target - cur
        objDistance = np.linalg.norm(vec)
        # print(cur)
        # print(target)
        # print(objDistance)
        if objDistance > 2 and "husky" in fixed:
              raise Exception("Husky can not move as it is on a stool")  
        cur = cur + (objDistance - tolerances[t]) * (vec / objDistance)
        x2, y2, z2 = target[0], target[1], target[2]
        x1, y1, z1, o1, done = cur[0], cur[1], z1, math.atan2((y2-y1),(x2-x1))%(2*math.pi), True
        # print(cur)

      elif(inpAction == "changeWing"):
        done = True

      elif(inpAction == "checkGrabbed"):
        if actions[action_index][1] not in constraints['ur5']:
          raise Exception("Object '" + actions[action_index][1] + "' not grabbed by the robot")
        done = True

      elif(inpAction == "constrain"):
        obj, t = actions[action_index][1], actions[action_index][2]
        target = np.array(metrics[obj][0]); cur = np.array([x1, y1, z1])
        vec = target - cur
        objDistance = np.linalg.norm(vec)
        # print(cur)
        # print(target)
        # print(objDistance)
        if not done:
          if t == 'ur5' and not "Movable" in properties[obj]:
              raise Exception("Object '" + obj + "' is not grabbable")
          if (t == 'ur5'
              and "Heavy" in properties[obj]
              and len(fcw(obj)) > 0
              and "Heavy" in properties[fcw(obj)[0]]):
              raise Exception("Robot can not hold stack of heavy objects")
          if len(constraints['ur5']) > 0 and t == 'ur5':
              raise Exception("Gripper is not free, can not hold object")
          if t == obj:
              raise Exception("Cant place object on itself")
          if (t == 'ur5' and cin(obj)):
              raise Exception("Object is inside an enclosure, can not grasp it.")
          if (t in enclosures and closed(t)):
              raise Exception("Enclosure is closed, can not place object inside")
          if (t == 'ur5' and objDistance > 2.95):
              raise Exception("Object too far away, move closer to it")
          if (t == 'ur5' and abs(metrics[obj][0][2] - z1 > 1.5)):
                raise Exception("Object on different height, please use stool/ladder")
          if ("mop" in obj
              or "sponge" in obj
              or "vacuum" in obj
              or "blow_dryer" in obj):
              cleaner = True
          if ("stick" in obj):
              stick = True
          if t not in constraints.keys():
            constraints[t] = []
          if fct(obj) != "":
            constraints[fct(obj)].remove(obj)
          constraints[t].append(obj)
          done = True

      elif(inpAction == "removeConstraint"):
        obj, t = actions[action_index][1], actions[action_index][2]
        if ("stick" in obj):
              stick = False
        if not done:
          cleaner = False
          constraints[t].remove(obj)
          done = True

      elif(inpAction == "changeState"):
        obj, state = actions[action_index][1], actions[action_index][2]
        if len(constraints['ur5']) > 0 and not stick:
            raise Exception("Gripper is not free, can not change state")
        if (state == 'stuck' and "Stickable" not in properties[obj]):
            raise Exception("Object not stickable")
        if state == 'on' or state == 'off':
          if (state == 'on'
            and "Can_Fuel" in properties[obj]
            and not obj in fueled):
            raise Exception("First add fuel to object and then switch on")
          if state in on:
            on.append(obj)
          else:
            if obj in on: on.remove(obj)
          done = True
        else:
          metrics[obj] = states[obj][state]
          done = True

      elif(inpAction == "climbUp"):
        t = actions[action_index][1]
        (x2, y2, z2), _ = metrics[t]
        height = 2 if t == 'ladder' else 0.4
        x1, y1, z1, o1, done = x2, y2, z2 + height, math.atan2((y2-y1),(x2-x1))%(2*math.pi), True
      
      elif(inpAction == "climbDown"):
        t = actions[action_index][1]
        (x2, y2, z2), _ = metrics[t]
        on_height = z1 > 0.5 and z1 < 1.1
        opposite = -1 if on_height else 1
        (x2, y2, z2) = (x2, y2+(opposite * 1.7 if y2 < 0 else -1.7 * opposite), 1 if on_height else 0)
        x1, y1, z1, o1, done = x2, y2, z2, math.atan2((y2-y1),(x2-x1))%(2*math.pi), True

      elif(inpAction == "clean"):
        obj = actions[action_index][1]
        if obj in clean:
            raise Exception("Object already clean")
        if not cleaner:
            raise Exception("No cleaning agent with the robot")
        if "Oily" in properties[obj] and 'blow_dryer' in constraints['ur5']:
            raise Exception("Can not clean oily substance with blow dryer")
        if 'blow_dryer' in constraints['ur5'] and not 'blow_dryer' in on:
            raise Exception("Please switch on blow dryer")
        clean.append(obj)
        done = True

      elif(inpAction == "addTo"):
        obj, lst = actions[action_index][1], actions[action_index][2]
        if lst == "sticky":
          if not "Stickable" in properties[obj]:
            raise Exception("Object is not stickable, cannot apply glue/tape agent")
          if obj not in sticky: sticky.append(obj) 
        elif lst == "fixed":
          if obj == "screw" and not "screwdriver" in constraints['ur5']:
              raise Exception("Driving a screw needs screwdriver")
          if obj == "screw" and not fct(obj) in drilled:
              raise Exception("Driving a screw needs object to be drilled first")
          if obj == "nail" and not ("hammer" in constraints['ur5'] or "brick" in constraints['ur5']):
              raise Exception("Driving a nail needs hammer or brick")
          if obj not in fixed: fixed.append(obj) 
        if lst == "drilled":
          if obj not in drilled: drilled.append(obj) 
        if lst == "welded":
          if fct(obj) != "assembly_station":
              raise Exception("First place object on assembly station")
          if obj not in welded: welded.append(obj) 
        if lst == "painted":
          if obj not in painted: painted.append(obj) 
        done = True 

      elif(inpAction == "fuel"):
        obj, fuel = actions[action_index][1], actions[action_index][2]
        if obj in fueled:
            raise Exception("Object has already been fueled")
        if not "Fuel" in properties[fuel]:
            raise Exception("Objects is not a fuel")
        if "Cuttable" in properties[fuel] and fuel not in cut:
            raise Exception("Object needs to be cut before being used as a fuel")
        if not "Can_Fuel" in properties[obj]:
            raise Exception("Can not fuel object " + obj)
        fueled.append(obj) 
        done = True 

      elif(inpAction == "cut"):
        obj, cutter = actions[action_index][1], actions[action_index][2]
        if obj in cut:
            raise Exception("Object has already been cut")
        if not "Cuttable" in properties[obj]:
            raise Exception("Objects " + obj + " is not cuttable")
        if not "Cutter" in properties[cutter]:
            raise Exception("Object " + cutter + " is not a cutter")
        cut.append(obj) 
        done = True 

      elif(inpAction == "print"):
        obj = actions[action_index][1]
        if obj in id_lookup.keys():
            raise Exception("Object already in world")
        object_list = []
        with open(object_file, 'r') as handle:
            all_objects = json.load(handle)['objects']
        for o in all_objects:
          if o['name'] == obj: break
        if not "Printable" in prop:
            raise Exception("Object can not be printed")
        metrics[obj] = [[-2.5, 4, 1.7], []]
        states[obj] = []
        tolerances[obj] = o['tolerance']
        properties[obj] = o['properties']
        ur5_dist[obj] = o['ur5_dist']
        cons_cpos_lookup[obj] = o['constraint_cpos']
        cons_pos_lookup[obj] = o['constraint_pos']
        done = True 
      
      elif(inpAction == "removeFrom"):
        obj, lst = actions[action_index][1], actions[action_index][2]
        if lst == "sticky":
          if obj in sticky: sticky.remove(obj) 
        elif lst == "fixed" and obj in fixed:
          if obj in fixed: fixed.remove(obj) 
        done = True 

      elif(inpAction == "saveBulletState"):
        done = True

      updateMetrics()

      if done:
        if not inpAction == "saveBulletState" and not "check" in inpAction:
          datapoint.addPoint([x1, y1, 0, o1], sticky, fixed, cleaner, actions[action_index], getGoalConstraints(), metrics, on, clean, stick, welded, drilled, painted, fueled, cut)
        action_index += 1
        done = False

def execute(actions, goal_file=None, queue_for_execute_to_stop = None, saveImg = True):
  global datapoint
  try:
    return executeHelper(actions, goal_file, queue_for_execute_to_stop, saveImg)
  except Exception as e:
    datapoint.addSymbolicAction("Error = " + str(e))
    datapoint.addPoint(None, None, None, None, 'Error = ' + str(e), None, None, None, None, None, None, None, None, None, None)
    raise e

def saveDatapoint(filename):
  global datapoint
  f = open(filename + '.datapoint', 'wb')
  pickle.dump(datapoint, f)
  f.flush()
  f.close()

def getDatapoint():
  return datapoint

def destroy():
  p.disconnect()
                      
def executeAction(inp):
    if execute(convertActionsFromFile(inp), args.goal, saveImg=False):
    	print("Goal Success!!!")
    else:
    	print("Goal Fail!!!")

def testPlan(domain, goal_num, world_num, plan):
  args = Args()
  args.world = 'jsons/'+ domain + '_worlds/world_' + domain + str(world_num) +'.json'
  args.goal = 'jsons/' + domain + '_goals/' + GOAL_LISTS[domain][goal_num - 1]
  plan = {'actions': plan}
  start(args)
  try:
    res = execute(plan, args.goal, saveImg=False)
    if res:
      return 1, 0, 0, ''
    return 0, 1, 0, ''
  except Exception as e:
    return 0, 0, 1, str(e)

def initPolicy(domain, goal_num, world_num):
  args = Args()
  args.world = 'jsons/'+ domain + '_worlds/world_' + domain + str(world_num) +'.json'
  args.goal = 'jsons/' + domain + '_goals/' + GOAL_LISTS[domain][goal_num - 1]
  start(args)

def execAction(goal_num, action, e):
  plan = {'actions': [action]}
  try:
    res = execute(plan, args.goal, saveImg=False)
    graph_data = datapoint.getGraph(embeddings = e)
    graph_data = graph_data["graph_"+str(len(graph_data)-1)] 
    g = convertToDGLGraph(graph_data, False, goal_num, -1)
    return res, g, ''
  except Exception as e:
    return False, None, str(e)

def printAllValues():
  print(sticky, fixed, on, fueled, cut, cleaner, stick, clean, drilled, welded, painted, metrics)

if __name__ == '__main__':
	# take input from user
	args = initParser()
	inp = args.input
	start(args)
	executeAction(inp)

	datapoint = getDatapoint()
	print(datapoint.toString(metrics=False))
	saveDatapoint('test')
