# RectAngle
Segmentation and classification tool for trans-rectal B-mode ultrasound images.

*Submitted as coursework for MPHY0041: Machine Learning in Medical Imaging.*

This package contains PyTorch-based implementations of a U-Net based segmentation model, and a **TBC** based classification model, for the simultaneous detection and segmentation of prostate in rectal b-mode ultrasound images.

**NEED TO WRITE UP REST OF DETAILS**

## Installation
A conda environment file is provided in the *./conda/* folder, which may be used as such:

>conda env create --file ./conda/rectangle.yml
>conda activate rectangle

Once this is activated, the package may be installed using the *setup.py* file:

>pip install .

Following this, training/inference may be performed from the command line using the scripts *train.py* and *predict.py* respectively.