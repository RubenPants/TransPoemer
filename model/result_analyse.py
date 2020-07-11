"""
result_analyse.py

Analyse the predictions of the model.
"""
import os

from utils import read_as_file

SNIPPET_SIZE = 15
SAMPLE = 'abracadabra'


# an example of limericks would say
# "i'm a man," said the "i'm a "a-a-a-a-a-a-a-a-a-a-a-d-a-d-a-d-a-d-d-a-d-d-a-d-d-
# the "abracadabra," said the "abracadabra, "i'm in the rain."

def analyse(text):
    print(f"Analyzing:  [{text}]")
    full_limericks = read_as_file(f'data/processed/full/complete')
    occ = 0
    snippet_occ = 0
    snippets = []
    for sample in full_limericks:
        # Full occurrence
        if text in sample: occ += 1
        
        # Snippet occurrence
        i = SNIPPET_SIZE
        while i < len(text):
            if text[i - SNIPPET_SIZE:i] in sample:
                snippets.append(text[i - SNIPPET_SIZE:i])
                snippet_occ += 1
                break
            i += SNIPPET_SIZE
    
    print(f"> The given sample occurred  {occ}  times in the full limericks corpus")
    print(f"> Snippets of length {SNIPPET_SIZE} occurred  {snippet_occ}  times in the full limericks corpus")
    if snippets:
        print(f">   {min(len(snippets), 3)} of these snippets:  {snippets[:3]}")


if __name__ == '__main__':
    os.chdir("..")
    analyse(SAMPLE)
