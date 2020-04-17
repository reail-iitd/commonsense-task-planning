import pybullet as p
import pybullet_data
import json
import os
import time
import numpy as np

def loadObject(name, position, orientation, obj_list):
    """
    Load an object based on its specified position and orientation
    Generate constraints for the object as specified
    :param: names, positions and orientation of objects
    :return: object index
    """
    urdf = ''
    obj = 0
    object_id = 0
    noise = np.zeros(3)
    for obj in obj_list:
      if obj['name'] == name:
        urdf = obj['urdf']
        if "noise" in obj['constraints']:
            noise = np.random.normal(0, size=3, scale=0.05)
            noise[-1] = 0
        break
    if orientation == []:
      object_id = p.loadURDF(urdf, list(position+noise))
    else:
      object_id = p.loadURDF(urdf, list(position+noise), orientation)
    return (object_id, 
            ("horizontal" in obj['constraints']), 
            ("on_ground" in obj['constraints']),
            ("fixed_orientation" in obj['constraints']),
            obj["tolerance"], 
            obj["properties"], 
            obj["constraint_cpos"], 
            obj["constraint_pos"], 
            obj["constraint_link"],
            obj["ur5_dist"])


def loadWorld(objects, object_file):
    """
    Load all objects specified in the world and create a user friendly dictionary with body
    indexes to be used by pybullet parser at the time of loading the urdf model. 
    :param objects: List containing names of objects in the world with positions and orientations.
    :return: Dictionary of object name -> object index and object index -> name
    """
    object_list = []
    horizontal = []
    ground = []
    object_lookup = {}
    id_lookup = {}
    properties_lookup = {}
    cons_cpos_lookup = {}
    cons_pos_lookup = {}
    cons_link_lookup = {}
    fixed_orientation = {}
    states = {}
    ur5_dist = {}
    tolerances = {}
    with open(object_file, 'r') as handle:
        object_list = json.load(handle)['objects']
    for obj in objects:
        (object_id, 
            horizontal_cons, 
            gnd,
            fix,
            tol, 
            prop,
            cpos,
            pos, 
            link, 
            dist) = loadObject(obj['name'], obj['position'], obj['orientation'], object_list)
        if horizontal_cons:
            horizontal.append(object_id)
        if gnd:
            ground.append(object_id)
        if fix:
            fixed_orientation[object_id] = p.getBasePositionAndOrientation(object_id)[1]
        object_lookup[object_id] = obj['name']
        properties_lookup[obj['name']] = prop
        cons_cpos_lookup[obj['name']] = cpos
        id_lookup[obj['name']] = object_id
        states[obj['name']] = obj['states']
        cons_pos_lookup[obj['name']] = pos
        cons_link_lookup[obj['name']] = link
        ur5_dist[obj['name']] = dist
        tolerances[obj['name']] = tol
        print(obj['name'], object_id)
    return object_lookup, id_lookup, horizontal, ground, fixed_orientation, tolerances, properties_lookup, cons_cpos_lookup, cons_pos_lookup, cons_link_lookup, ur5_dist, states

def initWingPos(wing_file):
    """
    Initialize wing poses of UR5 gripper
    """
    wings = dict()
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "robotiq_85_left_knuckle_joint"]
    with open(wing_file, 'r') as handle:
        poses = json.load(handle)["poses"]
        for pose in poses:
            wings[pose["name"]] = dict(zip(controlJoints, pose["pose"]))
    return wings

def initHuskyUR5(world_file, object_file):
    """
    Load Husky and Ur5 module
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    with open(world_file, 'r') as handle:
        world = json.load(handle)
    (object_lookup, 
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
        states) = loadWorld(world['entities'], object_file)
    base = id_lookup['husky']
    arm = id_lookup['ur5']
    return base, arm, object_lookup, id_lookup, horizontal_list, ground_list, fixed_orientation, tolerances, properties, cons_cpos_lookup, cons_pos_lookup, cons_link_lookup, ur5_dist, states
 