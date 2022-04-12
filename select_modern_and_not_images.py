import sys
import os
import random
import pandas as pd

if __name__ == '__main__':
    quadrant = sys.argv[1]
    folder1 = "./datasets/All/"+quadrant
    folder2 = "./datasets/No_face/testB/"+quadrant
    image_name1 = random.choice(os.listdir(folder1))
    image_name2 = random.choice(os.listdir(folder2))
    print("The chosen general image for this song is", image_name1)
    print("The chosen landscape image for this song is", image_name2)
