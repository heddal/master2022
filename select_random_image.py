import sys
import os
import random

if __name__ == '__main__':
    folder = sys.argv[1]+"/photo2art_hedda/test_latest/images"
    while True:
        image_name = random.choice(os.listdir(folder))
        if "real" in  image_name:
            continue
        print("The chosen image for this song is", image_name)
        break
