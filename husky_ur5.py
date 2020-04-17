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
from operator import sub
import math
import pickle

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

def start(input_args):
  # Initialize husky and ur5 model
  global husky,robotID, object_lookup, id_lookup, horizontal_list, ground_list,fixed_orientation,tolerances, properties,cons_cpos_lookup,cons_pos_lookup, cons_link_lookup,ur5_dist,states,wings,gotoWing,constraints,constraint,x1, y1, o1
  global imageCount,yaw,ims,dist,pitch,ax,fig,cam,camX, camY, world_states,id1, perspective, wall_id, datapoint
  global light, args, speed

  # Connect to Bullet using GUI mode
  light = p.connect(p.GUI)

  # Add input arguments
  args = input_args
  speed = args.speed

  if (args.logging or args.display=="both"):
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

  ( husky,
    robotID, 
    object_lookup, 
    id_lookup, 
    horizontal_list, 
    ground_list,
    fixed_orientation,
    tolerances, 
    properties,
    cons_cpos_lookup,
    cons_pos_lookup, 
    cons_link_lookup,
    ur5_dist,
    states) = initHuskyUR5(args.world, object_file)
  print ("The world file is", args.world)

  # Initialize dictionary of wing positions
  wings = initWingPos(wings_file)

  # Fix ur5 to husky
  cid = p.createConstraint(husky, -1, robotID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, ], [0., 0., -.2],
                           [0, 0, 0, 1])

  # Set small gravity
  p.setGravity(0,0,-10)

  # Initialize gripper joints and forces
  controlJoints, joints = initGripper(robotID)
  gotoWing = getUR5Controller(robotID)
  gotoWing(robotID, wings["home"])

  # Position of the robot
  x1, y1, o1 = 0, 0, 0
  constraint = 0

  # List of constraints with target object and constraint id
  constraints = dict()

  # Init camera
  imageCount = 0
  yaw = 50
  ims = []
  dist = 5
  pitch = -35.0

  # Start video recording
  p.setRealTimeSimulation(0) 
  ax = 0; fig = 0; cam = []
  if args.display:
        ax, cam = initDisplay("both")
  elif args.logging:
        fig = initLogging()
  camX, camY = 0, 0

  # Mention names of objects
  mentionNames(id_lookup)

  # Save state
  world_states = []
  id1 = p.saveState()
  world_states.append([id1, x1, y1, o1, constraints])
  print(id_lookup)
  print(fixed_orientation)

  # Check Logging
  if args.logging or args.display:
      deleteAll("logs")

  # Default perspective
  perspective = "tp"

  # Wall to make trasparent when camera outside
  wall_id = -1
  if 'home' in args.world:
    wall_id = id_lookup['walls']
  if 'factory' in args.world:
    wall_id = id_lookup['wall_warehouse']

  # Initialize datapoint
  datapoint = Datapoint()
  try:
      g = args.goal.split("/")[-1].split(".")[0]; w = args.world.split('/')[-1].split(".")[0]
  except:
      g = args.goal.split("/")[-1].split(".")[0]; w = args.world.split('/')[-1].split(".")[0]
  datapoint.world = w
  datapoint.goal = g
 
# Print manipulation region bounding boxes
# for obj in id_lookup.keys():
#   print(obj, p.getAABB(id_lookup[obj]))

def changeView(direction):
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, perspective, on
  camTargetPos = [x1, y1, 0]
  dist = dist - 0.5 if direction == "in" else dist + 0.5 if direction == "out" else dist
  yaw = yaw - 25 if direction == "left" else yaw + 25 if direction == "right" else yaw
  print(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos)
  perspective = "tp" if perspective == "fp" and direction == None else "fp" if direction == None else perspective
  lastTime, imageCount = saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, on)


def showObject(obj):
  global world_states, x1, y1, o1, imageCount, on
  if not obj in id_lookup.keys():
    raise Exception("Object not in world, print it first")
  ((x, y, z), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[obj])
  _, imageCount = saveImage(0, imageCount, 'fp', ax, math.atan2(y,x)%(2*math.pi), cam, 2, yaw, pitch, [x, y, z], wall_id, on)
  time.sleep(0.5)
  _, imageCount = saveImage(0, imageCount, 'fp', ax, math.atan2(y,x)%(2*math.pi), cam, 7, yaw, pitch, [x, y, z], wall_id, on)
  time.sleep(1)
  firstImage()

def undo():
  global world_states, x1, y1, o1, imageCount, constraints, on, datapoint
  datapoint.addSymbolicAction("Undo")
  datapoint.addPoint(None, None, None, None, 'Undo', None, None, None, None, None, None, None, None, None, None)
  x1, y1, o1, constraints, world_states = restoreOnInput(world_states, x1, y1, o1, constraints)
  _, imageCount = saveImage(0, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, on)

def firstImage():
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, on
  z = 1 if p.getBasePositionAndOrientation(id_lookup['husky'])[0][2] > 0.5 else 0
  camTargetPos = [x1, y1, z]
  _, imageCount= saveImage(-250, imageCount, perspective, ax, o1, cam, dist, 50, pitch, camTargetPos, wall_id, on)

keyboard = False

def executeHelper(actions, goal_file=None, queue_for_execute_to_stop = None, saveImg = True):
  global x1, y1, o1, world_states, dist, yaw, pitch, camX, camY, imageCount, cleaner, on, datapoint, clean, stick, keyboard, drilled, welded, painted, fueled, cut
  # List of low level actions
  datapoint.addSymbolicAction(actions['actions'])
  actions = convertActions(actions, args.world)
  print(actions)
  action_index = 0
  done = False; done1 = False
  waiting = False
  startTime = time.time()
  lastTime = startTime
  datapoint.addPoint([x1, y1, 0, o1], sticky, fixed, cleaner, 'Start', constraints, getAllPositionsAndOrientations(id_lookup), on, clean, stick, welded, drilled, painted, fueled, cut)

  # Start simulation
  if True:
      # start_here = time.time()
      counter = 0
      while(True):
          if queue_for_execute_to_stop is not None:
            try:
              stop = queue_for_execute_to_stop.get(block = False)
              return False
            except:
              pass
          counter += 1
          z = 1 if p.getBasePositionAndOrientation(husky)[0][2] > 0.5 else 0
          camTargetPos = [x1, y1, z]
          if (args.logging or args.display=="both") and (counter % COUNTER_MOD == 0) and (saveImg):
            # start_image = time.time()
            lastTime, imageCount = saveImage(lastTime, imageCount, "fp", ax, o1, cam, 3, yaw, pitch, camTargetPos, wall_id, on)
            # image_save_time = time.time() - start_image
            # print ("Image save time", image_save_time)
          # Move UR5 by keyboard
          # x1, y1, o1, keyboard = moveKeyboard(x1, y1, o1, [husky, robotID])
          # moveUR5Keyboard(robotID, wings, gotoWing)
          # x1, y1, o1, world_states = restoreOnKeyboard(world_states, x1, y1, o1)
          keepHorizontal(horizontal_list)
          keepOnGround(ground_list)
          keepOrientation(fixed_orientation)
          # dist, yaw, pitch, camX, camY = changeCameraOnKeyboard(dist, yaw, pitch, camX, camY)

          # start = time.time()
          p.stepSimulation() 
          # print ("Step simulation time ",time.time() - start) 
          # print(checkGoal(goal_file, constraints, states, id_lookup), constraints)

          if action_index >= len(actions):
            yaw = 180*(math.atan2(y1,x1)%(2*math.pi))/math.pi - 90
            lastTime, imageCount = saveImage(lastTime, imageCount, perspective, ax, o1, cam, dist, yaw, pitch, camTargetPos, wall_id, on)
            return checkGoal(goal_file, constraints, states, id_lookup, on, clean, sticky, fixed, drilled, welded, painted)

          if(actions[action_index][0] == "move"):
            if "husky" in fixed:
              raise Exception("Husky can not move as it is on a stool")    
            target = actions[action_index][1]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], target, keyboard, speed)

          elif(actions[action_index][0] == "moveZ"):
            if "husky" in fixed:
                  raise Exception("Husky can not move as it is on a stool") 
            if (actions[action_index][1][0] == -1.5
                and actions[action_index][1][1] == 1.5
                and actions[action_index][1][2] == 1):
                if (findConstraintTo('ramp', constraints) != 'floor_warehouse'):
                  raise Exception("Can not move up withuot ramp")
                if grabbedObj("stool", constraints):
                  if id_lookup["stool"] in ground_list: ground_list.remove(id_lookup["stool"])
            target = actions[action_index][1]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], target, keyboard, speed, up=True)

          elif(actions[action_index][0] == "moveTo"):
            if objDistance("husky", actions[action_index][1], id_lookup) > 2 and "husky" in fixed:
                  raise Exception("Husky can not move as it is on a stool")  
            if abs(p.getBasePositionAndOrientation(id_lookup[actions[action_index][1]])[0][2] - 
              p.getBasePositionAndOrientation(husky)[0][2]) > 1 and not stick:
                  raise Exception("Object on different height, please use stool")
            if (p.getBasePositionAndOrientation(id_lookup[actions[action_index][1]])[0][2] - 
              p.getBasePositionAndOrientation(husky)[0][2] < -0.7):
                  raise Exception("Object on lower level, please move down")
            target = actions[action_index][1]
            if target == 'door' or target == 'dumpster':
              if not done1:
                x1, y1, o1, done1 = moveTo(x1, y1, o1, [husky, robotID], id_lookup['floor'], 
                                        tolerances[target], 
                                        keyboard,
                                        speed, 0)
              else:
                x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], [2.9, 4.4, 0],
                                        keyboard,
                                        speed, 1, 0)
                done1 = not done
            else:
              x1, y1, o1, done = moveTo(x1, y1, o1, [husky, robotID], id_lookup[target], 
                                      tolerances[target], 
                                      keyboard,
                                      speed, 0)

          elif(actions[action_index][0] == "moveToXY"):
            if objDistance("husky", actions[action_index][1], id_lookup) > 2 and "husky" in fixed:
                  raise Exception("Husky can not move as it is on a stool")  
            target = actions[action_index][1]
            if target == 'door' or target == 'dumpster':
              if not done1:
                x1, y1, o1, done1 = moveTo(x1, y1, o1, [husky, robotID], id_lookup['floor'], 
                                        tolerances[target], 
                                        keyboard,
                                        speed, 0)
              else:
                x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], [2.9, 4.4, 0],
                                        keyboard,
                                        speed, 1, 0)
                done1 = not done
            else:
              x1, y1, o1, done = moveTo(x1, y1, o1, [husky, robotID], id_lookup[target], 
                                      tolerances[target], 
                                      keyboard,
                                      speed, 0.9 if actions[action_index][1] == 'cupboard' else 0)

          elif(actions[action_index][0] == "changeWing"):
            if time.time()-startTime > 1.8:
              done = True
            pose = actions[action_index][1]
            gotoWing(robotID, wings[pose])

          elif(actions[action_index][0] == "checkGrabbed"):
            if not grabbedObj(actions[action_index][1],constraints):
              raise Exception("Object '" + actions[action_index][1] + "' not grabbed by the robot")
            done = True

          elif(actions[action_index][0] == "constrain"):
            if time.time()-startTime > 1:
              done = True; waiting = False
            if not waiting and not done:
              bounding_box = p.getAABB(id_lookup[actions[action_index][1]])
              if actions[action_index][2] == 'ur5' and not "Movable" in properties[actions[action_index][1]]:
                  raise Exception("Object '" + action[action_index][1] + "' is not grabbable")
              if (actions[action_index][2] == 'ur5'
                  and "Heavy" in properties[actions[action_index][1]]
                  and len(findConstraintWith(actions[action_index][1], constraints)) > 0
                  and "Heavy" in properties[findConstraintWith(actions[action_index][1], constraints)[0]]):
                  raise Exception("Robot can not hold stack of heavy objects")
              if checkUR5constrained(constraints) and actions[action_index][2] == 'ur5':
                  raise Exception("Gripper is not free, can not hold object")
              if actions[action_index][2] == actions[action_index][1]:
                  raise Exception("Cant place object on itself")
              if (actions[action_index][2] == 'ur5'
                  and checkInside(constraints, states, id_lookup, actions[action_index][1], enclosures)):
                  raise Exception("Object is inside an enclosure, can not grasp it.")
              if (actions[action_index][2] in enclosures
                  and isClosed(actions[action_index][2], states, id_lookup)):
                  raise Exception("Enclosure is closed, can not place object inside")
              if (actions[action_index][2] == 'ur5' 
                  and(objDistance(actions[action_index][1], actions[action_index][2], id_lookup)) > 2):
                  raise Exception("Object too far away, move closer to it")
              if (actions[action_index][2] == 'ur5' and abs(p.getBasePositionAndOrientation(id_lookup[actions[action_index][1]])[0][2] - 
                  p.getBasePositionAndOrientation(husky)[0][2]) > 1.2):
                    raise Exception("Object on different height, please use stool/ladder")
              if ("mop" in actions[action_index][1] 
                  or "sponge" in actions[action_index][1] 
                  or "vacuum" in actions[action_index][1]
                  or "blow_dryer" in actions[action_index][1]):
                  cleaner = True
              if ("stick" in actions[action_index][1]):
                  stick = True
              if (('tray' in actions[action_index][2] or 'book' in actions[action_index][2])
                  and  max(map(sub, bounding_box[1], bounding_box[0])) > 0.5):
                  raise Exception("Object too big to be placed")
              cid = constrain(actions[action_index][1], 
                              actions[action_index][2], 
                              cons_link_lookup, 
                              cons_cpos_lookup,
                              cons_pos_lookup,
                              id_lookup,
                              constraints,
                              ur5_dist)
              constraints[actions[action_index][1]] = (actions[action_index][2], cid)
              waiting = True

          elif(actions[action_index][0] == "removeConstraint"):
            if ("stick" in actions[action_index][1]):
                  stick = False
            if time.time()-startTime > 1:
              done = True; waiting = False
            if not waiting and not done:
              cleaner = False
              removeConstraint(constraints, actions[action_index][1], actions[action_index][2])
              del constraints[actions[action_index][1]]
              waiting = True

          elif(actions[action_index][0] == "changeState"):
            if checkUR5constrained(constraints) and not stick:
                raise Exception("Gripper is not free, can not change state")
            state = actions[action_index][2]
            if (actions[action_index][2] == 'stuck' 
                and "Stickable" not in properties[actions[action_index][1]]):
                raise Exception("Object not stickable")
            # if state == "stuck" and not "board" in actions[action_index][1] and not actions[action_index][1] in sticky:
            #     removeConstraint(constraints, actions[action_index][1], "")
            #     del constraints[actions[action_index][1]]
            #     raise Exception("Object not sticky")  
            if actions[action_index][2] == 'on' or actions[action_index][2] == 'off':
              if (actions[action_index][2] == 'on'
                and "Can_Fuel" in properties[actions[action_index][1]]
                and not actions[action_index][1] in fueled):
                raise Exception("First add fuel to object and then switch on")
              if actions[action_index][1] in on:
                on.remove(actions[action_index][1])
              else:
                on.append(actions[action_index][1])
              done = True
            else:
              if actions[action_index][1] == "board" or 'paper' in actions[action_index][1]:
                p.changeDynamics(id_lookup[actions[action_index][1]], -1, mass=0)
              done = changeState(id_lookup[actions[action_index][1]], states[actions[action_index][1]][state]) 

          elif(actions[action_index][0] == "climbUp"):
            target = id_lookup[actions[action_index][1]]
            (x2, y2, z2), _ = p.getBasePositionAndOrientation(target)
            height = 2 if actions[action_index][1] == 'ladder' else 0.4
            targetLoc = [x2, y2, z2 + height]
            x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], targetLoc, keyboard, speed, tolerance=0.15, up=True)
          
          elif(actions[action_index][0] == "climbDown"):
            if not waiting and not done:
              target = id_lookup[actions[action_index][1]]
              (x2, y2, z2), _ = p.getBasePositionAndOrientation(target)
              on_height = p.getBasePositionAndOrientation(husky)[0][2] > 0.5 and p.getBasePositionAndOrientation(husky)[0][2] < 1.1
              opposite = -1 if on_height else 1
              targetLoc = [x2, y2+(opposite * 1.7 if y2 < 0 else -1.7 * opposite), 1 if on_height else 0]
              waiting = True
            else:
              x1, y1, o1, done = move(x1, y1, o1, [husky, robotID], targetLoc, keyboard, speed, up=True)
              if done: waiting = False

          elif(actions[action_index][0] == "clean"):
            if actions[action_index][1] in clean:
                raise Exception("Object already clean")
            if not cleaner:
                raise Exception("No cleaning agent with the robot")
            if ("Oily" in properties[actions[action_index][1]]
              and grabbedObj('blow_dryer', constraints)):
                raise Exception("Can not clean oily substance with blow dryer")
            if grabbedObj('blow_dryer', constraints) and not 'blow_dryer' in on:
                raise Exception("Please switch on blow dryer")
            p.changeVisualShape(id_lookup[actions[action_index][1]], -1, rgbaColor = [1, 1, 1, 0])
            clean.append(actions[action_index][1])
            done = True

          elif(actions[action_index][0] == "addTo"):
            obj = actions[action_index][1]
            if actions[action_index][2] == "sticky":
              if obj in sticky:
                  raise Exception("Object already sticky")
              if not "Stickable" in properties[actions[action_index][1]]:
                raise Exception("Object is not stickable, cannot apply glue/tape agent")
              sticky.append(obj) 
            elif actions[action_index][2] == "fixed":
              if obj in fixed:
                  raise Exception("Object already driven")
              if obj == "screw" and not grabbedObj("screwdriver",constraints):
                  raise Exception("Driving a screw needs screwdriver")
              if obj == "screw" and not findConstraintTo(obj, constraints) in drilled:
                  raise Exception("Driving a screw needs object to be drilled first")
              if obj == "nail" and not (grabbedObj("hammer",constraints) or grabbedObj("brick",constraints)):
                  raise Exception("Driving a nail needs hammer or brick")
              fixed.append(obj) 
            if actions[action_index][2] == "drilled":
              if obj in drilled:
                  raise Exception("Object already drilled")
              drilled.append(obj) 
            if actions[action_index][2] == "welded":
              if obj in welded:
                  raise Exception("Object already welded")
              if findConstraintTo(obj, constraints) != "assembly_station":
                  raise Exception("First place object on assembly station")
              welded.append(obj)
              horizontal_list.append(id_lookup[obj])
            if actions[action_index][2] == "painted":
              if obj in painted:
                  raise Exception("Object already painted")
              p.changeVisualShape(id_lookup[obj], -1, rgbaColor = [1, 0.4, 0.1, 1])
              painted.append(obj)
            done = True 

          elif(actions[action_index][0] == "fuel"):
            obj = actions[action_index][1]
            if obj in fueled:
                raise Exception("Object has already been fueled")
            if not "Fuel" in properties[actions[action_index][2]]:
                raise Exception("Objects is not a fuel")
            if ("Cuttable" in properties[actions[action_index][2]]
                and not actions[action_index][2] in cut):
                raise Exception("Object needs to be cut before being used as a fuel")
            if not "Can_Fuel" in properties[obj]:
                raise Exception("Can not fuel object " + obj)
            fueled.append(obj) 
            done = True 

          elif(actions[action_index][0] == "cut"):
            obj = actions[action_index][1]
            if obj in cut:
                raise Exception("Object has already been cut")
            if not "Cuttable" in properties[obj]:
                raise Exception("Objects " + obj + " is not cuttable")
            if not "Cutter" in properties[actions[action_index][2]]:
                raise Exception("Object " + actions[action_index][2] + " is not a cutter")
            cut.append(obj) 
            done = True 

          elif(actions[action_index][0] == "print"):
            obj = actions[action_index][1]
            if obj in id_lookup.keys():
                raise Exception("Object already in world")
            object_list = []
            with open(object_file, 'r') as handle:
                object_list = json.load(handle)['objects']
            (oid, horizontal_cons, gnd,
                fix, tol, prop, cpos, pos, link, d) = loadObject(obj, [-2.5, 4, 1.7], [], object_list)
            if not "Printable" in prop:
                raise Exception("Object can not be printed")
            if horizontal_cons: horizontal.append(oid)
            if gnd: ground.append(oid)
            if fix: fixed_orientation[oid] = p.getBasePositionAndOrientation(oid)[1]
            object_lookup[oid] = obj
            properties[obj] = prop
            cons_cpos_lookup[obj] = cpos
            id_lookup[obj] = oid
            states[obj] = []
            cons_pos_lookup[obj] = pos
            cons_link_lookup[obj] = link
            ur5_dist[obj] = d
            tolerances[obj] = tol
            print("Printed new object", obj, oid)
            done = True 
          
          elif(actions[action_index][0] == "removeFrom"):
            obj = actions[action_index][1]
            if actions[action_index][2] == "sticky":
              sticky.remove(obj) 
            elif actions[action_index][2] == "fixed":
              fixed.remove(obj) 
            done = True 

          elif(actions[action_index][0] == "saveBulletState"):
            id1 = p.saveState()
            world_states.append([id1, x1, y1, o1, constraints])
            done = True

          if done:
            startTime = time.time()
            if not actions[action_index][0] == "saveBulletState" and not "check" in actions[action_index][0]:
              datapoint.addPoint([x1, y1, 0, o1], sticky, fixed, cleaner, actions[action_index], constraints, getAllPositionsAndOrientations(id_lookup), on, clean, stick, welded, drilled, painted, fueled, cut)
            action_index += 1
            if action_index < len(actions):
              print("Executing action: ", actions[action_index])
            done = False
          # total_time_taken = time.time() - start_here
          # print ("Total", total_time_taken)
          # print ("Fraction", image_save_time/total_time_taken)
          # start_here = time.time()

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

if __name__ == '__main__':
	# take input from user
	args = initParser()
	inp = args.input
	start(args)
	executeAction(inp)

	datapoint = getDatapoint()
	print(datapoint.toString(metrics=False))
	saveDatapoint('test')
