import json

def convertActions(inp, world):
    action_list = []

    for high_level_action in inp['actions']:
        args = high_level_action['args']
        # print("Action", high_level_action['name'], args)
        if high_level_action['name'] == 'pickNplaceAonB':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveTo", args[1]],
                ["removeConstraint", args[0], "ur5"],
                ["constrain", args[0], args[1]]
            ])

        if high_level_action['name'] == 'changeWing':
            action_list.extend([
                ["changeWing", args[0]]
            ])

        elif high_level_action['name'] == 'moveAToB':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveTo", args[1]]
            ])

        elif high_level_action['name'] == 'push':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["move", args[1]],
                ["removeConstraint", args[0], "ur5"]
            ])

        elif high_level_action['name'] == 'pushTo':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveToXY", args[1]],
                ["removeConstraint", args[0], "ur5"]
            ])


        elif high_level_action['name'] == 'moveTo':
            action_list.extend([
                ["moveTo", args[0]]
            ])

        elif high_level_action['name'] == 'move':
            action_list.extend([
                ["move", args[0]]
            ])

        elif high_level_action['name'] == 'moveUp':
            action_list.extend([
                ["move", [0.5, -0.5, 0]],
                ["moveZ", [-1.5, 1.5, 1]]
            ])
        
        elif high_level_action['name'] == 'moveDown':
            action_list.extend([
                ["move", [-0.601, 0.603, 1]],
                ["moveZ", [0.5, -0.5, 0]]
            ])

        elif high_level_action['name'] == 'pick':
            action_list.extend([
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"]
            ])

        elif high_level_action['name'] == 'place':
            action_list.extend([
                ["moveTo", args[1]],
                ["changeWing", "up"],
                ["constrain", args[0], args[1]]
            ])

        elif high_level_action['name'] == 'dropTo':
            action_list.extend([["moveTo", args[1]], ["constrain", args[0], args[1]]])
        
        elif high_level_action['name'] == 'drop':
            action_list.append(["removeConstraint", args[0], "ur5"])

        elif high_level_action['name'] == 'changeState':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["changeState", args[0], args[1]]
            ])
        
        elif high_level_action['name'] == 'placeRamp':
            action_list.extend([
                ["moveTo", "ramp"],
                ["changeWing", "up"],
                ["constrain", "ramp", "ur5"],
                ["move", [0.5,-0.5,0]],
                ["constrain", "ramp", "floor_warehouse"]
            ])

        elif high_level_action['name'] == 'stick' and 'home' in world:
                action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["move", [-2,3,0]],
                ["constrain", args[0], args[1]],
                ["changeState", args[0], "stuck"]
            ])

        elif high_level_action['name'] == 'stick' and 'factory' in world:
                action_list.extend([
                ["moveTo", args[1]],
                ["checkGrabbed", args[0]],
                ["constrain", args[0], args[1]],
                ["changeState", args[0], "stuck"]
            ])

        elif high_level_action['name'] == 'fuel':
                action_list.extend([
                ["moveTo", args[0]],
                ["checkGrabbed", args[1]],
                ["fuel", args[0], args[1]]
            ])

        elif high_level_action['name'] == 'cut':
                action_list.extend([
                ["moveTo", args[0]],
                ["checkGrabbed", args[1]],
                ["cut", args[0], args[1]]
            ])

        elif high_level_action['name'] == 'print':
                action_list.extend([
                ["moveTo", "3d_printer"],
                ["print", args[0]]
            ])

        elif high_level_action['name'] == 'drive':
                action_list.extend([
                ["moveTo", args[0]],
                ["checkGrabbed", args[1]],
                ["changeWing", "up"],
                ["addTo", args[0], "fixed"]
            ])

        elif high_level_action['name'] == 'weld':
                action_list.extend([
                ["moveTo", args[0]],
                ["checkGrabbed", "welder"],
                ["changeWing", "up"],
                ["addTo", args[0], "welded"]
            ])

        elif high_level_action['name'] == 'paint':
                action_list.extend([
                ["moveTo", args[0]],
                ["checkGrabbed", "spraypaint"],
                ["changeWing", "up"],
                ["addTo", args[0], "painted"]
            ])

        elif high_level_action['name'] == 'drill':
                action_list.extend([
                ["moveTo", args[0]],
                ["checkGrabbed", "drill"],
                ["changeWing", "up"],
                ["addTo", args[0], "drilled"]
            ])

        elif high_level_action['name'] == 'apply':
                action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveTo", args[1]],
                ["removeConstraint", args[0], "ur5"],
                ["addTo", args[1], "sticky"]
            ])
        
        elif high_level_action['name'] == 'climbUp':
            action_list.extend([
                ["moveTo", args[0]],
                ["climbUp", args[0]],
                ["addTo", "husky", "fixed"]
            ])
        
        elif high_level_action['name'] == 'climbDown':
                action_list.extend([
                ["removeFrom", "husky", "fixed"],
                ["climbDown", args[0]]
            ])

        elif high_level_action['name'] == 'clean':
                action_list.extend([
                ["moveTo", args[0]],
                ["clean", args[0]]
            ])

        action_list.append(["saveBulletState"])

    return action_list

def convertActionsFromFile(action_file):
    inp = None
    with open(action_file, 'r') as handle:
        inp = json.load(handle)
    return(inp)