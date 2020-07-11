# TransPoemer

Transforming regular text to a poem (limerick) via the means of a Transformer model (neural network). The figure below
shows the general model overview. The model consists out of two steps:
1. Extract the keywords from the text on which the text is based on
2. *Translate* these keywords to a poem
<p align="center">
  <img src="https://github.com/RubenPants/TransPoemer/blob/master/data/images/model.png"/>
</p>

Aside from the task to translate regular text into a poem, this repository investigates if it's possible to train such a
network on a laptop, requiring significantly less computational power than related state-of-the-art Transformer models.

## Report and Presentation
This project was part of a school project. The report as well as the final presentation can be found in the root 
directory; `TransPoemer_report.pdf` and `TransPoemer_presentation.pptx` respectively.

## Project
All functionality implemented in this project is handled via `main.py` (root directory).