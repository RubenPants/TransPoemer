"""
infer.py

Perform inference on the trained model.
"""
import argparse
import os

from data.process import extract_keywords
from model.setup import main as setup
from tokenizers.tokenizer_shared import poem_transform_keywords, display_attention
from transformer.seq2seq import Seq2Seq

# Parameters
DUMMY_INPUT = [
    # "black",
    "rain",
    # "example",
    # "corona",
    "",
    "bird nest children dog",
]


def main(model: Seq2Seq,
         inp: str,
         extract=True,
         visualize=True,
         ):
    """
    Perform inference on the model.
    
    :param model: The model which is used for the inference
    :param inp: Input sentence which will be poem'd
    :param extract: Extract the keywords from the input
    :param visualize: Visualize the attention layer
    """
    print(f"> Input:              {inp}")
    keywords = extract_keywords([inp])[0] if extract else inp
    print(f"> Extracted keywords: {keywords}")
    keywords_enc = model.tokenizer.tokenize(keywords)
    pred, attention = poem_transform_keywords(
            keywords=keywords_enc,
            model=model,
    )
    pred_stitch = model.tokenizer.stitch(pred[:-1])
    print(f"> Predicted poem:     {pred_stitch}")
    
    if visualize:
        display_attention(
                sentence=[model.tokenizer.albert.decode(t) for t in model.tokenizer.tokenize(keywords)],
                translation=[model.tokenizer.albert.decode(t) for t in pred[:-1]],
                attention=attention,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--model_name", type=str, default='model_pre')
    parser.add_argument("--input", type=list, default=DUMMY_INPUT)
    args = parser.parse_args()
    
    # Go back to root directory
    os.chdir("..")
    
    # Load the model and tokenizer
    mdl = setup(
            model_name=args.model_name,
    )
    
    # Evaluate the model
    for sample in args.input:
        main(
                model=mdl,
                inp=sample,
                extract=False,
        )

"""
--> model_1 (without pre-training) <--
> Input:              black
> Extracted keywords: black
> Predicted poem:     the black-blooded yanks would be found in the sky, and the black-and-white scene, and the black-and-white scene, and the "sigh" "the "s", "i'm ad say, "no, "boo!"
> Input:              rain
> Extracted keywords: rain
> Predicted poem:     the "the rain" is the sun, and the sun's wet. it's a clear demonstration of rain, and the sun's a good place, and the sun's got a good place to behold.
> Input:              example
> Extracted keywords: example
> Predicted poem:     an example of "a-"," said the "sew, "i'm a "s-ta-," i say, "i'm a "s-," "s-," i'm a "s-," "i'm",
> Input:              corona
> Extracted keywords: corona
> Predicted poem:     the coronary arteries, the corona, and the sun, and the sun, and the sun, and the sun, and the sun, and the corona are quite well known as a "swine!"
> Input:
> Extracted keywords:
> Predicted poem:     the "the "s" is the "s". "i'm a "s-a-"." "i'm a "s-uh-uh-uh-uh-uh-uh-uh-uh-uh-uh-uh-uh-uh-uh-uh
> Input:              bird nest children dog
> Extracted keywords: bird nest children dog
> Predicted poem:     a bird of the nest in the children, a dog-like bird, and the dog-footed wings. it's a dog-like bird, and the dog-eaters, and the dog is the best thing.



--> model_2 (with pre-training) <--
> Input:              black
> Extracted keywords: black
> Predicted poem:     the black-eyed young lady named (a "a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-k."
> Input:              rain
> Extracted keywords: rain
> Predicted poem:     the "the "old" is a "the """""" "it's a """""" "it's a "no," "it's a "no," "it's a "it"."
> Input:              example
> Extracted keywords: example
> Predicted poem:     an example of a "a"," said the "a"." "i'm a "a"." "it's a "a"." "it's a "a"." "it's a "a"."
> Input:              corona
> Extracted keywords: corona
> Predicted poem:     the corona's a wonderful place. the "the swine" is apt to be. "it's a "a"." "it's a "no," "no," "no," "no," "no,"
> Input:
> Extracted keywords:
> Predicted poem:     "i'm a man," said the guy, "i'm a man." "i'm a man," said the guy, "i'm a guy," said the guy. "i'm a guy." "i'm a guy."
> Input:              bird nest children dog
> Extracted keywords: bird nest children dog
> Predicted poem:     a bird's a nest, and the children are not a dog. it's a big deal, but it's a big deal. it's a big deal, but it's a big deal.



--> model_pre (only pre-training) <--
> Input:              black
> Extracted keywords: black
> Predicted poem:     the black is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is black.
> Input:              rain
> Extracted keywords: rain
> Predicted poem:     the rain is a beautiful thing.
> Input:              example
> Extracted keywords: example
> Predicted poem:     the example of the human race is that the most important thing is to be a man.
> Input:              corona
> Extracted keywords: corona
> Predicted poem:     the great thing about the great things that are currently living in the rainforest.
> Input:
> Extracted keywords:
> Predicted poem:     the best way to get is to be a good person.
> Input:              bird nest children dog
> Extracted keywords: bird nest children dog
> Predicted poem:     the bird is a nest of children, and the dog is a man.

"""
