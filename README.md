# ReRe:GP
Sound and Music Computing Masters Project, generating loopable symbolic music using TransformerXL and DadaGP dataset

## How to Use
Install dependencies
```
python -m pip install -r requirements.txt
```
Modify paths and other configuration details for training and inference in `full-data-config_5_lat1024.yml`

Run training
```
python train.py
```

Generate outputs (without extracting loops)
```
python inference.py
cd data_parse
python convert_folder ./inference_attempts/yyyymmdd-hhmmss OUTPUT FOLDER
```

**Generate ouputs and extract loops:** run `data_parse/extract_ex.ipynb` notebook

## Repo Layout

#### full-data-config_5_lat1024.yml
main config file with all the parameters for generation and training

#### model_ead.py
script for model backbone, adapted from  https://github.com/YatingMusic/compound-word-transformer

#### modules.py
script for model backbone, adapted from https://github.com/YatingMusic/compound-word-transformer

#### model weights/
folder where model weights and model config file should go

#### data/
folder for storing mappings between token strings and integer IDs, as well as dataset in npz format

#### [data_parse/](https://github.com/Satrat/ReRe-GP/tree/main/data_parse)
Scripts loop extraction, file format conversion, and generation examples

#### inference.py
Script for generation. Depends on the main config file, vocab pickle files, and npz dataset

#### train.py
Script for training. Depends on the main config file, vocab pickle files, and npz dataset

## Files that need to be downloaded externally
* ```data/fulldataset-song-artist-train_data_XL.npz```
* ```model_weights/ep_40.pth.tar```
