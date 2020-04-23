#!/usr/bin/env python3
from importlib import import_module
import os
import json
from flask import Flask, render_template, Response, request
from src.camera import Camera
from src.base_camera import BaseCamera
import multiprocessing as mp
import time
from src.parser import *
from os import listdir
import time
import random

args = initParser()

queue_from_webapp_to_simulator = mp.Queue()
queue_from_simulator_to_webapp = mp.Queue()
queue_for_error = mp.Queue()
queue_for_execute_to_stop = mp.Queue()
queue_for_execute_is_ongoing = mp.Queue()
workerId = None
base_url = ""
app = Flask(__name__)
moves_to_show = []

# All actions that need to be showed on the user screen.
dict_of_predicates = json.load("jsons/predicates_for_webapp.json")["dict_of_predicates"]

# Mapping of user action to the simulator action.
dict_predicate_to_action = json.load("jsons/predicates_for_webapp.json")["dict_predicate_to_action"]

# The list of goals that are possible for the simulator to execute.
GOAL_LIST = json.load("jsons/predicates_for_webapp.json")["GOAL_LIST"]

# The list of world instances that can be loaded by the simulator.
WORLD_LIST = json.load("jsons/predicates_for_webapp.json")["WORLD_LIST"]

# Load all objects reqiured
d = json.load(open(args.world))["entities"]
world_objects = []
renamed_objects = {}
constraints_dict = json.load(open("jsons/constraints.json"))
dropdown_states = ["open", "close", "off", "on", "up", "down"]
for obj in d:
    if (("ignore" in obj) and (obj["ignore"] == "true")):
        continue
    if ("rename" in obj):
        world_objects.append(obj["rename"])
        renamed_objects[obj["rename"]] = obj["name"]
    else:
        world_objects.append(obj["name"])
if '3d_printer' in world_objects:
    with open('jsons/objects.json') as file:
        o = json.load(file)["objects"]
    for obj in o:
        if 'Printable' in obj['properties'] and not obj['name'] in world_objects:
            world_objects.append(obj['name'])
world_objects.sort()

def convertActionsFromFile(action_file):
    inp = None
    with open(action_file, 'r') as handle:
        inp = json.load(handle)
    return(inp)

def simulator(queue_from_webapp_to_simulator, queue_from_simulator_to_webapp, queue_for_error, queue_for_execute_to_stop, queue_for_execute_is_ongoing):
    """
        The simulator loop accepting inputs from the user and sending it to the simulator.
        Also sends exception the web app for showing.
    """

    import husky_ur5
    import src.actions
    import sys
    husky_ur5.start(args)
    queue_from_simulator_to_webapp.put(True)
    print ("Waiting")
    husky_ur5.firstImage()
    goal_file = None
    while True:
        inp = queue_from_webapp_to_simulator.get()
        if ("rotate" in inp or "zoom" in inp or "toggle" in inp):
            husky_ur5.changeView(inp["rotate"])
        elif "undo" in inp:
            husky_ur5.undo()
            if (len(moves_to_show) > 0):
                moves_to_show.pop(-1)
        elif "showObject" in inp:
            try:
                husky_ur5.showObject(inp["showObject"])
            except Exception as e:
                print(e)
                queue_for_error.put(str(e))
        elif "restart" in inp:
            goal_file = inp["restart"]
            if (args.randomize and goal_file == None):
                args.goal = random.choice(GOAL_LIST)
                goal_file = args.goal
                args.world = random.choice(WORLD_LIST)
            elif (not args.randomize and goal_file == None):
                goal_file = args.goal
            else:
                args.world = random.choice(WORLD_LIST)
            husky_ur5.destroy()
            del sys.modules["husky_ur5"]
            del sys.modules["src.actions"]
            import husky_ur5
            import src.actions
            husky_ur5.start(args)
            husky_ur5.firstImage()
            queue_from_simulator_to_webapp.put(True)
        else:
            try:
                queue_for_execute_is_ongoing.put(True)
                done = husky_ur5.execute(inp, goal_file, queue_for_execute_to_stop)
                try:
                    queue_for_execute_is_ongoing.get(block=False)
                except:
                    pass
                print("Done: ", done)
            except Exception as e:
                print (str(e))
                queue_for_error.put(str(e))
                done = False
            if (done):
                w = 'factory' if 'factory' in args.world else 'home' if 'home' in args.world else 'outdoor'
                # foldername = 'dataset/home/' + goal_file.split("\\")[3].split(".")[0] + '/' + args.world.split('\\')[3].split(".")[0]
                try: 
                    foldername = 'dataset/' + w + '/' + goal_file.split("\\")[3].split(".")[0] + '/' + args.world.split('\\')[3].split(".")[0]
                except:
                    foldername = 'dataset/' + w + '/' + goal_file.split("/")[-1].split(".")[0] + '/' + args.world.split('/')[-1].split(".")[0]
                try:   
                    a = len(listdir(foldername))
                except Exception as e:
                    os.makedirs(foldername)
                if len(listdir(foldername)) == 0:
                    husky_ur5.saveDatapoint(foldername + '/' + '0')
                else:    
                    husky_ur5.saveDatapoint(foldername + '/' + str(a))
                queue_for_error.put("You have completed this tutorial.")
                queue_from_webapp_to_simulator.put({"restart": args.goal})
            called_undo_before = False

@app.route('/', methods = ["GET"])
def index():
    if args.randomize:
        goal = random.choice(GOAL_LIST)
        queue_from_webapp_to_simulator.put({"restart": goal})
    else:
        queue_from_webapp_to_simulator.put({"restart": None})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    if (request.method == "GET"):
        if args.randomize:
            return render_template('index.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects, base_url = base_url, goal_text = json.load(open(goal, "r"))["text"])
        else:
            return render_template('index.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects, base_url = base_url)
@app.route('/tutorial/1', methods = ["GET"])
def show_tutorial1():
    return render_template('tutorial1.html')

@app.route('/tutorial/2', methods = ["GET"])
def show_tutorial2():
    queue_from_webapp_to_simulator.put({"restart": None})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial2.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/tutorial/3', methods = ["GET"])
def show_tutorial3():
    return render_template('tutorial3.html')

@app.route('/tutorial/4', methods = ["GET"])
def show_tutorial4():
    queue_from_webapp_to_simulator.put({"restart": "jsons/home_goals/goal0-tut1.json"})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial4.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/tutorial/5', methods = ["GET"])
def show_tutorial5():
    return render_template('tutorial5.html')

@app.route('/tutorial/6', methods = ["GET"])
def show_tutorial6():
    queue_from_webapp_to_simulator.put({"restart": "jsons/home_goals/goal0-tut2.json"})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial6.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/tutorial/7', methods = ["GET"])
def show_tutorial7():
    return render_template('tutorial7.html')

@app.route('/tutorial/8', methods = ["GET"])
def show_tutorial8():
    queue_from_webapp_to_simulator.put({"restart": "jsons/home_goals/goal0-tut3.json"})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial8.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/workerId', methods = ["POST"])
def addworkerid():
    global workerId
    workerId = request.form["workerId"]
    print (workerId)
    return ""
    # show_tutorial1()
    # return render_template('index.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route("/arguments")
def return_arguments_for_predicate():
	text = request.args.get('predicate')
	return render_template("arguments.html", arguments_list = list(enumerate(dict_of_predicates[text].items())), world_objects = world_objects, constraints_dict = constraints_dict[text], dropdown_states = dropdown_states)
    
def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/simulator_state')
def get_simulator_state():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/restart_process')
def restart_process():
    try:
        queue_for_execute_is_ongoing.get(block = False)
        queue_for_execute_to_stop.put(True)
        time.sleep(1)
        return "restarted_process_successfully"
    except:
        return "did_not_need_to_restart"

@app.route("/execute_move", methods = ["POST"])
def execute_move():
    print (request.form)
    predicate = request.form["predicate"]
    l = []
    front_end_objects = []
    i = 0
    while True:
        if ("arg" + str(i) in request.form):
            front_end_object = request.form["arg" + str(i)]
            front_end_objects.append(front_end_object)
            if front_end_object in renamed_objects:
                l.append(renamed_objects[front_end_object])
            else:
                l.append(front_end_object)
            i += 1
        else:
            break
    if "Ramp" in predicate or "ramp" in predicate:
        l = []; front_end_objects = []
    d = {
        'actions': [
        {
            'name': str(dict_predicate_to_action[predicate]),
            'args': list(l)
        }
        ]
    }
    print (d)
    if len(front_end_objects) > 0:
        move_string = predicate + " ( " + str(front_end_objects[0])
    else:
        move_string = predicate + " ( "
    for i in range(1,len(front_end_objects)):
        move_string += " ," + str(front_end_objects[i])
    move_string += " )"
    print (move_string)
    moves_to_show.append(move_string)
    queue_from_webapp_to_simulator.put(d)
    return move_string

@app.route("/showObject", methods = ["POST"])
def showObject():
    object_to_show = request.form["object"]
    if object_to_show in renamed_objects:
        object_to_show = renamed_objects[object_to_show]
    print (object_to_show)
    queue_from_webapp_to_simulator.put({"showObject": object_to_show})
    return ""

@app.route("/rotateCameraLeft", methods = ["POST"])
def rotateCameraL():
    queue_from_webapp_to_simulator.put({"rotate": "left"})
    return ""

@app.route("/rotateCameraRight", methods = ["POST"])
def rotateCameraR():
    queue_from_webapp_to_simulator.put({"rotate": "right"})
    return ""

@app.route("/zoomIn", methods = ["POST"])
def zoomIn():
    queue_from_webapp_to_simulator.put({"rotate": "in"})
    return ""

@app.route("/zoomOut", methods = ["POST"])
def zoomOut():
    queue_from_webapp_to_simulator.put({"rotate": "out"})
    return ""

@app.route("/toggle", methods = ["POST"])
def toggle():
    queue_from_webapp_to_simulator.put({"rotate": None})
    return ""

@app.route("/undo_move", methods = ["GET"])
def undo_move():
    queue_from_webapp_to_simulator.put({"undo": True})
    return ""

@app.route("/check_error", methods = ["GET"])
def is_error():
    try:
        err_string = queue_for_error.get(block = False)
        return err_string
    except:
        return ""

if __name__ == '__main__':
    inp = "jsons/input_home.json"
    p = mp.Process(target=simulator, args=(queue_from_webapp_to_simulator,queue_from_simulator_to_webapp,queue_for_error, queue_for_execute_to_stop, queue_for_execute_is_ongoing))
    p.start()
    should_webapp_start = queue_from_simulator_to_webapp.get()
    # The ip address where to host the simulator can be changed here.
    app.run(host='0.0.0.0', threaded=True)
