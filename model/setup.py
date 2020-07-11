"""
setup.py

Build and initialize the model and the tokenizer.
"""
import argparse
import os.path

import torch

from transformer.seq2seq import count_parameters, get_cuda_device, initialize_weights, Seq2Seq
from utils import drop, make_folder, prep

# Parameters
VERSION = 0  # Version of the model, 0 if not versioned  [def=0]
HID_DIM = 128  # Dimension of the hidden layer  [def=128]
ENC_LAYERS = 3  # Number of layers/head for the encoder  [def=3]
DEC_LAYERS = 3  # Number of layers/head for the decoder  [def=3]
ENC_HEADS = 8  # Number of heads in the encoder  [def=8]
DEC_HEADS = 8  # Number of heads in the decoder  [def=8]
ENC_PF_DIM = 256  # Point-wise Feedforward dimension in the encoder  [def=256]
DEC_PF_DIM = 256  # Point-wise Feedforward dimension in the decoder  [def=256]
ENC_DROPOUT = 0.1  # Dropout ratio in the encoder  [def=0.1]
DEC_DROPOUT = 0.1  # Dropout ratio in the decoder  [def=0.1]


def main(model_version=0,
         model_name='',
         overwrite=False,
         hid_dim=HID_DIM,
         enc_layers=ENC_LAYERS,
         dec_layers=DEC_LAYERS,
         enc_heads=ENC_HEADS,
         dec_heads=DEC_HEADS,
         enc_pf_dim=ENC_PF_DIM,
         dec_pf_dim=DEC_PF_DIM,
         enc_dropout=ENC_DROPOUT,
         dec_dropout=DEC_DROPOUT,
         ):
    """
    Create and initialize the model and tokenizer.
    
    :param model_version: Version of the model, 0 if not versioned
    :param model_name: Name of the model [optional]
    :param overwrite: Overwrite if model already exists
    :param hid_dim: Dimension of the hidden layer
    :param enc_layers: Number of layers/head for the encoder
    :param dec_layers: Number of layers/head for the decoder
    :param enc_heads: Number of heads in the encoder
    :param dec_heads: Number of heads in the decoder
    :param enc_pf_dim: Point-wise Feedforward dimension in the encoder
    :param dec_pf_dim: Point-wise Feedforward dimension in the decoder
    :param enc_dropout: Dropout ratio in the encoder
    :param dec_dropout: Dropout ratio in the decoder
    :return: model, tokenizer
    """
    prep("Creating skeleton of the model...\n", key="model")
    model_name = model_name if model_name else f'model{f"_{model_version}" if model_version else ""}'
    model = Seq2Seq(
            name=model_name,
            overwrite=overwrite,
            hid_dim=hid_dim,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            enc_heads=enc_heads,
            dec_heads=dec_heads,
            enc_pf_dim=enc_pf_dim,
            dec_pf_dim=dec_pf_dim,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
    ).to(get_cuda_device())
    drop(key="model")
    print(f' --> The model has {count_parameters(model):,} trainable parameters\n')
    
    if not overwrite and os.path.isfile(f'model/models/{model}.pt'):
        # Load previously stored model
        model.load_state_dict(torch.load(f'model/models/{model}.pt'))
        print(f"Successfully loaded the previously stored model: '{model}'!")
    else:
        # Initialize the model with randomized weights
        model.apply(initialize_weights)
        
        # Save current state of model
        make_folder('model/models')
        torch.save(model.state_dict(), f'model/models/{model}.pt')
        print(f"Model '{model}' saved!")
    
    # Setup all the right folders
    make_folder(f'model/models')
    make_folder(f'model/models/{model}')
    make_folder(f'model/models/{model}/images')
    if not os.path.isfile(f'model/models/{model}/log.txt'):
        with open(f'model/models/{model}/log.txt', 'w') as f:
            f.write('')  # Create log-file
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--version", type=int, default=VERSION)
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument('--hid_dim', type=int, dest="hid", default=HID_DIM)
    parser.add_argument('--enc_layers', type=int, default=ENC_LAYERS)
    parser.add_argument('--dec_layers', type=int, default=DEC_LAYERS)
    parser.add_argument('--enc_heads', type=int, default=ENC_HEADS)
    parser.add_argument('--dec_heads', type=int, default=DEC_HEADS)
    parser.add_argument('--enc_pf_dim', type=int, dest="enc_pf", default=ENC_PF_DIM)
    parser.add_argument('--dec_pf_dim', type=int, dest="dec_pf", default=DEC_PF_DIM)
    parser.add_argument('--enc_dropout', type=int, default=ENC_DROPOUT)
    parser.add_argument('--dec_dropout', type=int, default=DEC_DROPOUT)
    args = parser.parse_args()
    
    # Go back to root directory
    os.chdir("..")
    
    main(
            model_version=args.version,
            overwrite=args.overwrite,
            hid_dim=args.hid,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            enc_heads=args.enc_heads,
            dec_heads=args.dec_heads,
            enc_pf_dim=args.enc_pf,
            dec_pf_dim=args.dec_pf,
            enc_dropout=args.enc_dropout,
            dec_dropout=args.dec_dropout,
    )
