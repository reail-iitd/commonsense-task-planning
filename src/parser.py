import argparse

def initParser():
    """
    Input arguments for the simulation
    """
    parser = argparse.ArgumentParser('This will simulate a world describe in a json file.')
    parser.add_argument('--world', 
                            type=str,
                            default = "jsons/home_worlds/world_home0.json",
                            help='The json file to visualize')
    parser.add_argument('--input',
                            type=str,
                            required=False,
                            default="jsons/input.json",
                            help='The json file of input high level actions')
    parser.add_argument('--logging',
                            type=bool,
                            required=False,
                            default=False,
                            help='Video recording of simulation')
    parser.add_argument('--display',
                            type=str,
                            required=False,
                            default="both",
                            help='Display states on matplotlib animation')
    parser.add_argument('--speed',
                            type=float,
                            required=False,
                            default=1.0,
                            help='How quickly to step through the visualization')
    parser.add_argument('--goal',
                            type=str,
                            required=False,
                            default='jsons/home_goals/goal0-tut1.json',
                            help='Path of goal file')
    parser.add_argument('--randomize',
                            action = "store_true",
                            help = "Turning this on randomizes the goal file and the scene file")
    return parser.parse_args()
 