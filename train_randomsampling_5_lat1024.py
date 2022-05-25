
import os
import json
import yaml
import pickle
import datetime
import numpy as np
from collections import OrderedDict

import torch
#from model_randomsampling import TransformerXL
from model_ead import TransformerXL


def main():
    # gen config
    modelConfig, trainConfig = get_configs()

    # load dictionary
    event2word = pickle.load(open("vocab_song_artist.pkl", 'rb')) # fulldataset non-splitted
    print("event2word size: ", len(event2word))
    word2event = pickle.load(open("rev_vocab_song_artist.pkl", 'rb'))
    print("word2event size: ", len(word2event))

    # load train data
    training_data = np.load(os.path.join('','fulldataset-song-artist-train_data_XL.npz'))

    device = torch.device("cuda" if not trainConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = trainConfig['gpuID']

    print('Device to train:', device)
    
    resume = trainConfig['resume_training_model']

    # declare model
    model = TransformerXL(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=True)

    # train
    model.train(training_data,
                trainConfig,
                device,
                resume)
            

def get_configs():
    cfg = yaml.full_load(open("full-data-config_5_lat1024.yml", 'r')) 

    modelConfig = cfg['MODEL']
    trainConfig = cfg['TRAIN']

    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_Dir = os.path.join(trainConfig['output_dir'],"5_lat1024" + cur_date)
    if not os.path.exists(experiment_Dir):
        print('experiment_Dir:', experiment_Dir)
        os.makedirs(experiment_Dir) 
    print('Experiment: ', experiment_Dir)
    trainConfig.update({'experiment_Dir': experiment_Dir})


    with open(os.path.join(experiment_Dir, 'full-data-config.yml'), 'w') as f:
        doc = yaml.dump(cfg, f)

    print('='*5, 'Model configs', '='*5)
    print(json.dumps(modelConfig, indent=1, sort_keys=True))
    print('='*2, 'Training configs', '='*5)
    print(json.dumps(trainConfig, indent=1, sort_keys=True))
    return modelConfig, trainConfig


if __name__ == '__main__':
    main()


