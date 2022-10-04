# MVEN

Dataset and code for the paper "M3: An Emotion and Novelty-aware Approach for Multilingual Multimodal Misinformation Detection"

General Instructions -

* Before running any code, we have to create the conda environment using environment.yml file.

'''
conda env create -f environment.yml
'''

* For running the model 
* * with full dataset (Hindi + Tamil + Bangla) use - python python-filename full
* * with hindi dataset use - python python-filename h
* * with bangla dataset use - python python-filename b
* * with tamil dataset use - python python-filename t

------------------------------------------------------------------------

In code directory there are 5 subdirectories

* Final_Data: 
It has all the processed data, ready to be fed in model.

* Source Information collection: It has all code to collect background information. Please run the file according to serial number given in file.

* data:
It has all the additional data like mediabias.json etc and data created during building model.

* emotion:
THis folder is for computing the emotion of target image.

* models:
This folder contains the code for the main model architecture.

NOTE: Everywhere in this repository *nv* means *novelty* and *em* means *emotion*
