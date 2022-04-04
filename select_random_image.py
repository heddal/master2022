import sys
import os
import random
from PIL import Image

if __name__ == '__main__':
    folder = sys.argv[1]
    image_name = random.choice(os.listdir(folder))
    image = Image.open(folder + '/' + image_name)
    image.show()
