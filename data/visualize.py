"""
visualize.py

Visualizations for a list of poems to do a manual analysis on.
"""
import argparse
import collections
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from transformers import AlbertTokenizer

from utils import read_as_file


def token_occurrence(poems, tokenizer, data_tag='', tok_tag='', save=True):
    """ Number of word (lower-case) occurrences in the data. Words will be filtered based on the SentenceTokenizer. """
    # Counter
    print("Encoding and counting")
    token_count = collections.Counter()
    for segment in poems:
        encoded = tokenizer.encode(segment, add_special_tokens=False)  # numerical value (encoding)
        for e in encoded:
            token_count[e] += 1
    
    # Number of unique tokens available in the dataset
    print(f"--> {len(token_count)} unique tokens")
    
    # Check how many tokens are only used once
    singleton = 0
    for k in token_count.keys():
        if token_count[k] == 1:
            singleton += 1
    print(f"--> Number of singletons: {singleton}")
    
    # Check how many tokens are only used twice
    double = 0
    for k in token_count.keys():
        if token_count[k] == 2:
            double += 1
    print(f"--> Number of doubles: {double}")
    
    # Check how many tokens are only used three times
    triples = 0
    for k in token_count.keys():
        if token_count[k] == 3:
            triples += 1
    print(f"--> Number of triples: {triples}")
    
    # Check how many tokens are used less or equal than ten times
    less = 0
    for k in token_count.keys():
        if token_count[k] <= 10:
            less += 1
    print(f"--> Number of less or equal than 10: {less}")
    
    # Rearrange counter based on value
    print("Rearranging")
    token_count_sorted = {k: v for k, v in sorted(token_count.items(), key=lambda item: -item[1])}  # Greatest first
    
    # Visualizer
    print("Visualizing")
    create_bar_plot(token_count_sorted,
                    sort=False,
                    prune=0.7,
                    save_path=f'data/images/token_count_{data_tag.split(" ")[0]}.png' if save else None,
                    title=f'Syllable count: {data_tag} - {tok_tag}',
                    x_label='token count',
                    y_label='token occurrence')


def sequence_length(poems, tokenizer, data_tag='', tok_tag='', save: bool = True):
    """
    Visualize the average length for each of the sequences. Note that sequences get clipped on length 512 (can be turned
    off if requested).
    """
    # Counting
    counter = collections.Counter()
    for segment in poems:
        encoded = tokenizer.encode(segment, add_special_tokens=False)
        counter[len(encoded)] += 1
    
    # Filling in the gaps
    max_key = max(counter.keys())
    for k in range(max_key):
        if k not in counter:
            counter[k] = 0
    
    # Visualizing
    create_bar_plot(counter,
                    prune=1,
                    save_path=f'data/images/sequence_length_{data_tag.split(" ")[0]}.png' if save else None,
                    title=f'Sequence length: {data_tag} - {tok_tag}',
                    x_label='sequence length',
                    y_label='number of sequences')


def create_bar_plot(d: dict, prune=0.95, sort=True, save_path=None, title='', x_label=None, y_label=None):
    """
    Create a bar-plot for the given dictionary.

    :param d: Dictionary that must be plotted
    :param prune: Display only the first _ percentage of the plot
    :param sort: Sort the dictionary based on key-values
    :param save_path: None: do not save | String: save the plot under the given path
    :param title: Title of the plot
    :param x_label: Label of the x-axis
    :param y_label: label of the y-axis
    """
    # Sort on value (increasing)
    if sort:
        keys, values = zip(*sorted(zip(d.keys(), d.values())))
    else:
        values = list(d.values())
        keys = list(range(len(values)))
    
    # Maximum values on the respective axis
    x_max = len(values) - 1  # Since key 0 was also taken in consideration
    y_max = max(values)
    
    # Only visualize first <prune> percent of samples (exclude outliers)
    if type(values[0]) == int:
        total = sum(values)
        keep = round(total * prune)
        index = len([i for i in range(x_max + 1) if sum(values[:i]) <= keep])
        if index == 1:
            index += 1
        values = values[:index]
        keys = keys[:index]
        if (x_max + 1) != len(values):
            title += ' - first ' + str(round(prune * 100)) + '%'
        y_label = y_label + ' - max: ' + str(y_max) if y_label else 'max: ' + str(y_max + 1)
    if not keys or type(keys[0]) == int:
        x_label = x_label + ' - max: ' + str(x_max) if x_label else 'max: ' + str(x_max + 1)
    
    ax = plt.figure(figsize=(8, 4)).gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    width = 1 if len(keys) > 50 else 0.8  # 0.8 is default
    plt.bar(range(len(values)), list(values), width=width)
    if keys:
        key_length = len(keys)
        step = key_length // 50 + 1  # Max 50 labels on x-axis
        plt.xticks(range(0, key_length, step), list(keys[0::step]), rotation='vertical')
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def main(path,
         data_tag,
         tok_tag="albert-base-v1",
         token_count=True,
         seq_length=True,
         save=True,
         ):
    """
    Create visualizations for the processed data.
    
    :param path: Relative path towards the data
    :param data_tag: Tag used for saving and title generation
    :param tok_tag: Pre-trained tokenizer's tag
    :param token_count: Create visualization for the token-count
    :param seq_length: Create visualization for the sequence-length
    :param save: Save the result of the image
    """
    # Load in the tokenizer
    tok = AlbertTokenizer.from_pretrained(tok_tag)
    
    # Visualizations
    dataset = read_as_file(path)
    if token_count:
        token_occurrence(dataset, data_tag=data_tag, tokenizer=tok, tok_tag=tok_tag, save=save)
    if seq_length:
        sequence_length(dataset, data_tag=data_tag, tokenizer=tok, tok_tag=tok_tag, save=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--tok', type=str, default='albert-base-v1')  # pre-trained tokenizer's tag
    parser.add_argument('--path', type=str, default='processed/backup/combined')  # Path of the data
    parser.add_argument('--tag', type=str, default='combined')  # Tag used for saving and title generation
    parser.add_argument('--token_count', type=bool, dest='token', default=True)  # Time consuming visualization!
    parser.add_argument('--sequence_length', type=bool, dest='sequence', default=True)
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()
    
    # Go back to root directory
    os.chdir("..")
    
    # Visualize
    main(
            tok_tag=args.tok,
            path=args.path,
            data_tag=args.tag,
            token_count=args.token,
            seq_length=args.sequence,
            save=args.save,
    )
