"""
evaluate.py

Evaluate the trained model.
"""
import argparse
import math
import os

import torch.nn as nn

from model.setup import main as setup
from transformer.seq2seq import calculate_gleu, evaluate
from utils import drop, prep


def main(model):
    """ Evaluate the model based on the GLEU metric. """
    prep("Calculating the test-loss...\n", key="test")
    criterion = nn.CrossEntropyLoss(ignore_index=model.trg_pad_idx)
    test_loss = evaluate(model, model.tokenizer.test_iterator, criterion)
    drop(key="test")
    print(f'Test Loss: {test_loss:.3f}')
    
    prep("Calculating the GLEU score...", key="gleu")
    gleu = calculate_gleu(
            model=model,
            data=model.tokenizer.test[:1000],  # TODO: Takes too long otherwise!
    )
    drop(key="gleu")
    print(f'GLEU score = {gleu}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--version", type=int, default=0)
    args = parser.parse_args()
    
    # Go back to root directory
    os.chdir("..")
    
    # Setup the model
    mdl = setup(
            model_version=args.version,
    )
    
    # Evaluate the model
    main(
            model=mdl,
    )
