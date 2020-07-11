"""
train.py

Train the transformer on our custom dataset, mapping keywords to complete sentences.
"""
import argparse
import os
import time
from random import sample

import pandas as pd
import torch
import torch.nn as nn

from model.setup import main as setup
from transformer.seq2seq import calculate_gleu, epoch_time, evaluate, Seq2Seq, train
from utils import drop, make_folder, prep

# Parameters
LR = 0.0005  # The learning rate of the training
STAG = 3  # Number of stagnated training iterations before termination


def execute_training(model: Seq2Seq,
                     tokenizer,
                     learning_rate=LR,
                     epochs=10,
                     model_prefix='',
                     ):
    """
    Train the model on our custom dataset, mapping keywords to complete sentences.

    :param model: The model that must be trained
    :param tokenizer: Tokenizer to provide the data
    :param learning_rate: Learning rate of the training
    :param epochs: Number of epochs the training will last
    :param model_prefix: Prefix added when saving the model
    """
    prep("Initializing training...", key="init")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.trg_pad_idx)
    drop(key="init")
    
    make_folder(f'model/models/{model}/')
    
    print("Start training:")
    best_loss = float("inf")
    stagnation = 0
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, tokenizer.train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, tokenizer.valid_iterator, criterion)
        end_time = time.time()
        
        # Display intermediate stats
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Determine the gleu-score on the validation set, perform on 200 random sampled valid samples (performance)
        sampled = sample(list(tokenizer.valid), 200)
        gleu_score = calculate_gleu(model=model, data=sampled)
        
        # Print out results
        print(f'> Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} ')
        print(f'\t----------------- | Val. GLEU: {gleu_score}')
        
        # Hard early stopping: Save as long as the validation loss decreases
        torch.save(model.state_dict(), f'model/models/{model}/{model_prefix}{model}_epoch{epoch + 1:02d}.pt')
        print(f'\t --> Model saved under "{model_prefix}{model}_epoch{epoch + 1:02d}.pt" in model\'s folder')
        if valid_loss < best_loss:
            stagnation = 0
            best_loss = valid_loss
            torch.save(model.state_dict(), f'model/models/{model}.pt')
            print(f'\t --> Model improved: saved under "{model}.pt"')
        else:
            stagnation += 1
            if stagnation >= STAG:
                print("\t --> Training has stagnated, terminating")
                break
            else:
                print("\t --> Model did not improve!")


def train_quotes(model, epochs=10):
    """ Pre-train the transformer on the target task (sentence prediction based on keywords), using quotes. """
    execute_training(
            model=model,
            tokenizer=model.tokenizer_pre,
            learning_rate=LR,
            epochs=epochs,
            model_prefix='pre_'
    )


def train_limericks(model, epochs=20):
    """ Train the Transformer on the limericks dataset. """
    execute_training(
            model=model,
            tokenizer=model.tokenizer,
            learning_rate=LR,
            epochs=epochs,
    )


def main(model,
         quotes=False,
         quotes_epochs=5,
         limericks=False,
         limericks_epochs=20,
         ):
    """
    Perform the training.
    
    :param model: Model that must be trained
    :param quotes: Perform pre-training on a keyword-to-sentence translation task on the quotes dataset
    :param quotes_epochs: Epochs of training for the quotes dataset
    :param limericks: Perform training on a keyword-to-sentence translation task on the limericks dataset
    :param limericks_epochs: Epochs of training for the limericks dataset
    """
    if quotes:
        train_quotes(
                model=model,
                epochs=quotes_epochs,
        )
    if limericks:
        if quotes:
            # Load most fit version of the model again
            model = setup(
                    model_name=str(model),
            )
        train_limericks(
                model=model,
                epochs=limericks_epochs,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--pre_train", dest="pre", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=True)
    args = parser.parse_args()
    
    # Set wider run-terminal
    desired_width = 320
    pd.set_option('display.width', desired_width)
    
    # Go back to root directory
    os.chdir("..")
    
    # Create the model
    mdl: Seq2Seq = setup(
            model_version=args.version,
    )
    
    # Train the model
    main(
            model=mdl,
            quotes=args.pre,
            limericks=args.train,
    )
