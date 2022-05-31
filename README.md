# DadaGP Generation
WIP to generate loopable phrases using TransformerXL and DadaGP dataset

## Layout

#### full-data-config_5_lat1024.yml
main config file with all the parameters for generation and training

#### model_ead.py
script for model backbone, adapted from  https://github.com/YatingMusic/compound-word-transformer

#### modules.py
script for model backbone, adapted from https://github.com/YatingMusic/compound-word-transformer

#### model weights/
folder where model weights should go, also model config file

#### data/
folder for storing mappings between token strings and integer IDs, as well as dataset in npz format

#### util/
Scripts for converting between DadaGP and GuitarPro formats. Adapted from https://github.com/dada-bots/dadaGP

#### inference.py
Script for generation. Depends on the main config file, vocab pickle files, and npz dataset

#### train.py
Script for training. Depends on the main config file, vocab pickle files, and npz dataset

## Files that need to be downloaded externally
* ```data/fulldataset-song-artist-train_data_XL.npz```
* ```model_weights/ep_200.pth.tar```
