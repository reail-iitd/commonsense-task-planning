import time
from src.base_camera import BaseCamera
import os

class Camera(BaseCamera):
    """An emulated camera implementation that streams a repeated sequence of
    files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""

    @staticmethod
    def frames():
        # open_dir = "logs/tp/"
        # contents = os.listdir(open_dir)

        # while "0.jpg" not in contents:
        #     contents = os.listdir(open_dir)
        # print ("hi")
        # os.remove(open_dir + "0.jpg")
        # x = 1
        # imgs = {}
        # while True:
        #     time.sleep(0.3)
        #     contents = os.listdir(open_dir)
        #     for i in contents:
        #         if i not in imgs:
        #             imgs[i] = open(open_dir + i, "rb").read()
        #     if (str(x) + ".jpg" in contents and (str(x) + ".jpg" in imgs)):
        #         print ("hello" + str(x))
        #         yield imgs[str(x) + ".jpg" ]
        #         del imgs[str(x) + ".jpg" ]
        #         # time.sleep(0.1)
        #         x += 1
        #     if "0.jpg" in contents:
        #         print ("no")
        #         os.remove(open_dir + "0.jpg")
        #         x = 1   
        outer_dir = "logs/"
        x = 0
        while True:
            time.sleep(0.1)
            if (os.path.exists(outer_dir + str(x) + ".jpg")):
                yield open(outer_dir + str(x) + ".jpg", "rb").read()
            else:
                x = 0
                if (os.path.exists(outer_dir + str(x) + ".jpg")):
                    yield open(outer_dir + str(x) + ".jpg", "rb").read()
            if (os.path.exists(outer_dir + str(x+1) + ".jpg")):
                x += 1