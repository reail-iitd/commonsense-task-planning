import pybullet as p
import math
from scipy.spatial import distance

sign = lambda x: x and (1, -1)[x < 0]

def move(x1, y1, o1, object_list, target_coordinates, keyboard, speed, tolerance=0, up=False):
    """
    Move robot towards target coordinate location
    :params: 
        x1 - current x coordinate of objects in object_list
        y1 - current y coordinate of objects in object_list
        o1 - current angle of objects in object_list
        object_list - list of object ids to be moved
        target_coordinates - coordinates of target location
        keyboard - if currently moving via keyboard
        speed - speed of motion
        tolerance - how close to reach target location
        up - move along z axis or not
    :return:
        x1 - updated x coordinate of objects in object_list
        y1 - updated y coordinate of objects in object_list
        o1 - updated angle of objects in object_list
        moved - move operation complete or not
    """
    if keyboard:
        return x1, y1, o1, False
    delz = 0
    (x1, y1, z1) = p.getBasePositionAndOrientation(object_list[0])[0]
    x2 = target_coordinates[0]; y2 = target_coordinates[1]; z2 = target_coordinates[2]
    robot, dest = o1%(2*math.pi), math.atan2((y2-y1),(x2-x1))%(2*math.pi)
    left = (robot - dest)%(2*math.pi); right = (dest - robot)%(2*math.pi)
    dist = abs(distance.euclidean((x1, y1, z1), (x2, y2, z2)))
    if dist > 0.3 and left > 0.05 and right > 0.05:
        o1 = o1 + 0.004*speed if left > right else o1 - 0.004*speed 
    elif dist > tolerance + 0.1: 
        x1 += math.cos(o1)*0.008*speed
        y1 += math.sin(o1)*0.008*speed
        delz = 0.008*speed*sign(z2-z1) if up else 0
    else:
        return x1, y1, o1, True
    q=p.getQuaternionFromEuler((0,0,o1))
    for obj_id in object_list:
        (x, y, z1) = p.getBasePositionAndOrientation(obj_id)[0]
        p.resetBasePositionAndOrientation(obj_id, [x1, y1, z1+delz], q)
    return x1, y1, o1, False


def moveTo(x1, y1, o1, object_list, target, tolerance, keyboard, speed, offset):
    """
    Move robot towards a target object
    :params: 
        x1 - current x coordinate of objects in object_list
        y1 - current y coordinate of objects in object_list
        o1 - current angle of objects in object_list
        object_list - list of object ids to be moved
        target - object id of target to which the objects need to be moved to
        tolerance - tolerance distance of the target object
        keyboard - if currently moving via keyboard
        speed - speed of motion
    :return:
        x1 - updated x coordinate of objects in object_list
        y1 - updated y coordinate of objects in object_list
        o1 - updated angle of objects in object_list
        moved - move operation complete or not
    """
    if keyboard:
        return x1, y1, o1, False
    y2 = p.getBasePositionAndOrientation(target)[0][1] + offset
    x2 = p.getBasePositionAndOrientation(target)[0][0]
    z2 = p.getBasePositionAndOrientation(target)[0][2]
    target_coordinates = [x2, y2, z2]
    husky = object_list[0]
    if ((target_coordinates[2] >= 1.8 and p.getBasePositionAndOrientation(husky)[0][2] <= 1.0) or
        (target_coordinates[2] <= 1.4 and p.getBasePositionAndOrientation(husky)[0][2] >= 1.8)):
        raise Exception("Target object is not on same level, please first move to the same level as target")
    return move(x1, y1, o1, object_list, target_coordinates, keyboard, speed, tolerance)


def constrain(obj1, obj2, link, cpos, pos, id_lookup, constraints, ur5_dist):
    """
    Constrain two objects
    :params: 
        obj1 - object to be constrained
        obj2 - target object to which obj1 is constrained
        link - link lookup for objects to be constrained
        id_lookup - id dictionary to lookup object id by name
        constraints - current list of constraints
        ur5_dist - dictionary to lookup distance from ur5 gripper
    :return:
        cid - constraint id
    """
    if obj1 in constraints.keys():
        p.removeConstraint(constraints[obj1][1])
    count = 0 # count checks where to place on target object
    for obj in constraints.keys():
        if constraints[obj][0] == obj2:
            count += 1
    print("New constraint=", obj1, " on ", obj2)
    # parent is the target, child is the object
    if obj2 == "ur5":
        cid = p.createConstraint(id_lookup[obj2], link[obj2], id_lookup[obj1], link[obj1], p.JOINT_POINT2POINT, [0, 0, 0], 
                                parentFramePosition=ur5_dist[obj1],
                                childFramePosition=cpos[obj1][0],
                                childFrameOrientation=[0,0,0,0])
    else:
        cid = p.createConstraint(id_lookup[obj2], link[obj2], id_lookup[obj1], link[obj1], p.JOINT_POINT2POINT, [0, 0, 0], 
                                parentFramePosition=pos[obj2][count],
                                childFramePosition=cpos[obj1][0],
                                childFrameOrientation=[0,0,0,0])
    return cid

def removeConstraint(constraints, obj1, obj2):
    """
    Remove constraint between two objects
    :params: 
        constraints - current dictionary of constraints
        obj1 - constrained object
        obj2 - target object to which obj1 is constrained
    """
    if obj1 in constraints.keys():
        p.removeConstraint(constraints[obj1][1])

def changeState(obj, positionAndOrientation):
    """
    Change state of an object
    :params: 
        obj - if of object
        positionAndOrientation - target state of object
    :return:
        done - if object state is very close to target state
    """
    q=p.getQuaternionFromEuler(positionAndOrientation[1])
    ((x1, y1, z1), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(obj)
    ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
    done = True
    x1 = x1 + 0.01*sign(x2-x1); done = done and abs(x2-x1) <= 0.01
    y1 = y1 + 0.01*sign(y2-y1); done = done and abs(y2-y1) <= 0.01
    z1 = z1 + 0.01*sign(z2-z1); done = done and abs(z2-z1) <= 0.01
    a1 = a1 + 0.01*sign(a2-a1); done = done and abs(a2-a1) <= 0.01
    b1 = b1 + 0.01*sign(b2-b1); done = done and abs(b2-b1) <= 0.01
    c1 = c1 + 0.01*sign(c2-c1); done = done and abs(c2-c1) <= 0.01
    d1 = d1 + 0.01*sign(d2-d1); done = done and abs(d2-d1) <= 0.01
    p.resetBasePositionAndOrientation(obj, (x1, y1, z1), (a1, b1, c1, d1))
    return done
