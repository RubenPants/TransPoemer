"""
visualize.py

Visualize the inference progress of the model by means of the model's attention layers.
"""
import argparse
import os

from model.result_analyse import analyse
from model.setup import main as setup
from tokenizers.tokenizer_pre import TokenizerPre
from tokenizers.tokenizer_shared import display_attention
from transformer.seq2seq import gleu_score, poem_transform_keywords, Seq2Seq
from utils import make_folder, save_as_json


def main(model: Seq2Seq,
         src_id,
         src,
         trg,
         save=True,
         analysis=False,
         ):
    """
    Visualize the attention layers of each head during inference of the model.
    
    :param model: The model on which the inference is performed
    :param src_id: ID of the input
    :param src: The input sequence on which the inference is performed
    :param trg: The target output
    :param save: Save the result
    :param analysis: Do a small analysis on the result
    """
    src_stitched = model.tokenizer.stitch(src)
    trg_stitched = model.tokenizer.stitch(trg)
    print(f"\nCreating visualization for: {src_id}")
    print(f'> src = {src_stitched}')
    print(f'> trg = {trg_stitched}')
    
    translation, attention = poem_transform_keywords(
            keywords=src,
            model=model,
    )
    pred_stitched = model.tokenizer.stitch(translation[:-1])
    gleu = gleu_score([[translation[:-1]]], [trg])
    print(f'> predicted trg = {pred_stitched}')
    print(f'> GLEU score: {gleu}')
    
    # Add the analysis of the result to it
    if analysis: analyse(pred_stitched)
    
    # Save prediction
    if save:
        save_path = f"model/models/{model}/images/{src_id}"
        make_folder(save_path)
        j = {
            'src':  src_stitched,
            'trg':  trg_stitched,
            'pred': pred_stitched,
            'gleu': gleu
        }
        save_as_json(j, f'{save_path}/prediction')
    else:
        save_path = None
    
    # Visualize each of the attention layers
    src_dec = [model.tokenizer.albert.decode(s) for s in src]
    translation_dec = [model.tokenizer.albert.decode(t) if type(t) == int else t for t in translation]
    display_attention(
            sentence=src_dec,
            translation=translation_dec,
            attention=attention,
            save_path=save_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--version", type=int, default=0)
    args = parser.parse_args()
    
    # Go back to root directory
    os.chdir("..")
    
    # Visualize the prediction
    mdl: Seq2Seq = setup(
            model_name='model_pre_10',  # TODO
            # model_version=args.version,
    )
    
    # Load in other tokenizer
    tok = TokenizerPre()
    for example_idx in range(5):
        s = tok.test.examples[example_idx].src
        t = tok.test.examples[example_idx].trg
        main(
                model=mdl,
                src_id=f"pre_{example_idx}",
                src=s,
                trg=t,
        )
