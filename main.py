"""
main.py

Run the full model-training and evaluation pipeline. Note that you must've run the processing of the data first!
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch

from data.process import main as process_data
from model.evaluate import main as evaluate
from model.setup import main as setup
from model.train import main as train
from model.visualize import main as visualize
from transformer.seq2seq import Seq2Seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--version", type=bool, default=1)
    parser.add_argument("--overwrite", type=bool, default=False)
    # Data
    parser.add_argument("--process_data", dest="process", type=bool, default=False)
    parser.add_argument("--process_raw", dest="raw", type=bool, default=False)
    # Training
    parser.add_argument("--pre_train", dest="quotes", type=bool, default=False)
    parser.add_argument("--train", dest="train", type=bool, default=False)
    # Evaluation
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=True)
    args = parser.parse_args()
    
    # Planting the seeds
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    # Set wider run-terminal
    desired_width = 320
    pd.set_option('display.width', desired_width)
    
    # Process the raw data
    if args.process:
        process_data(
                raw=args.raw,
                limericks=True,
                quotes=True,
        )
    
    # Setup the model
    model: Seq2Seq = setup(
            # model_name='model_pre',  # TODO
            overwrite=args.overwrite,
            model_version=args.version,
    )
    
    # Train the model on our own dataset (limericks)
    if args.train or args.quotes:
        train(
                model=model,
                quotes=args.quotes,
                quotes_epochs=5,
                limericks=args.train,
                limericks_epochs=20,
        )
        
        # Load most fit version of the model again
        if args.evaluate or args.visualize:
            model = setup(
                    model_name=str(model),
            )
    
    # Evaluate the model
    if args.evaluate:
        evaluate(
                model=model,
        )
    
    # Visualize inference example
    if args.visualize:
        for example_idx in range(50):
            src = model.tokenizer.test.examples[example_idx].src
            trg = model.tokenizer.test.examples[example_idx].trg
            visualize(
                    model=model,
                    src_id=f"test_sample_{example_idx}",
                    src=src,
                    trg=trg,
                    # save=False,  # TODO
                    analysis=False,
            )
