# RectAngle
Segmentation and classification tool for trans-rectal B-mode ultrasound images.

*Submitted as coursework for MPHY0041: Machine Learning in Medical Imaging.*

This package contains PyTorch-based implementations of a U-Net based segmentation model, and a DenseNet-based classification model, for the simultaneous detection and segmentation of prostate in rectal b-mode ultrasound images.

**NEED TO WRITE UP REST OF DETAILS**

## Installation

To install from command line, use the following git command:

>git clone https://github.com/liamchalcroft/RectAngle/

A conda environment file is provided in the *./conda/* folder, which may be used as such:

>cd RectAngle

>conda env create --file ./conda/rectangle.yml

>conda activate rectangle

Once this is activated, the package may be installed using the *setup.py* file:

>pip install .

Following this, training/inference may be performed from the command line using the scripts *train.py* and *predict.py* respectively.

An interactive notebook used for experiments in the associated report is available:

<a href="https://colab.research.google.com/github/liamchalcroft/RectAngle/blob/main/colab_from_package.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
