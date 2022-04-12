import sys
import os
import random
import pandas as pd

if __name__ == '__main__':
    folder = sys.argv[1]
    a = 0
    b = 0
    wikiart_info = pd.read_csv('data/WikiArt-Ag4-cleaned.tsv', sep='\t')
    correct_style = wikiart_info.loc[wikiart_info['Style']
                                     == 'Post Renaissance Art']
    while a == 0 or b == 0:
        image_name = random.choice(os.listdir(folder))
        if (correct_style['Title'].apply(lambda x: True if correct_style['Image URL'].str.contains(image_name).any() else False).any()) and a == 0:
            print("The chosen post renaissance image for this song is", image_name)
            a += 1
        elif b == 0:
            print("The chosen general image for this song is", image_name)
            b += 1
