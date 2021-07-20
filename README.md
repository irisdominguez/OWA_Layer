# OWA_Layer
Implementation of the OWA layer in Fastai.

This repository contains the code associated with our latest experiments, detailed in a paper to be published. The bibliographic reference will be added here as soon as it is available.

## Requirements

This experiments were run in a conda environment created according to:

```
channels:
  - pytorch
  - fastai
  - anaconda
  - defaults
  - conda-forge
dependencies:
  - cudatoolkit
  - fastai::fastai=1.0.61
  - ffmpeg
  - ipykernel
  - ipywidgets
  - jupyterlab_widgets
  - matplotlib
  - numpy
  - conda-forge::opencv
  - pandas
  - python
  - pytorch-gpu
  - scikit-learn
  - torchvision
  - seaborn
```

## Usage

The code is centered around a reference pseudoexperiment and three main experiments with OWA layers, which can be run secuentially from their associated scripts:

```
python exp_reference.py
python exp_init.py
python exp_feats.py
python exp_metrics.py
```

From there, a jupyter notebook (`analyze_experiment.ipynb`) processes the results and presents both the tables and figures from the paper.

## Documentation
The code is still largely uncommented. We will try to improve the legibility with time, but for now, feel free to email me with any question.

Thank you for your understanding.
