import sys
import os
import random
import pandas as pd

if __name__ == '__main__':
    folder = sys.argv[1]
    a = 0
    b = 0
    wikiart_info = pd.read_csv('data/WikiArt-Ag4-cleaned.tsv', sep='\t')
    while a == 0 or b == 0:
        image_name = random.choice(os.listdir(folder))
        print("Looking at image",image_name)
        if ((wikiart_info['Title'] == image_name) & (wikiart_info['Style'] == 'Modern Art')).any() and a < 1:
            print("The chosen modern image for this song is", image_name)
            a += 1
        elif ((wikiart_info['Title'] == image_name) & (wikiart_info['Style'] != 'Modern Art')).any() and b < 1:
            print("The chosen non-modern image for this song is", image_name)
            b += 1
        print(a,",",b)
