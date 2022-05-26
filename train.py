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

import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    modelConfig, trainConfig = get_configs(rank)
    dist.barrier()

    # load dictionary
    event2word = pickle.load(open("vocab_song_artist.pkl", 'rb'))
    word2event = pickle.load(open("rev_vocab_song_artist.pkl", 'rb'))

    # load train data
    training_data = np.load(trainConfig['data_path'])
    
    # load trained model config
    resume = trainConfig['resume_training_model']

    # declare model
    model = TransformerXL(
            modelConfig,
            rank,
            event2word=event2word, 
            word2event=word2event, 
            is_training=True)

    # train
    model.train(training_data,
                trainConfig,
                resume)

    cleanup()

def get_configs(rank):
    cfg = yaml.full_load(open("full-data-config_5_lat1024.yml", 'r')) 

    modelConfig = cfg['MODEL']
    trainConfig = cfg['TRAIN']

    if rank == 0:
        cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        experiment_Dir = os.path.join(trainConfig['output_dir'],"5_lat1024" + cur_date)
        if not os.path.exists(experiment_Dir):
            print('Creating experiment_dir:', experiment_Dir)
            os.makedirs(experiment_Dir) 
        print('Experiment: ', experiment_Dir)
        trainConfig.update({'experiment_dir': experiment_Dir})

        with open(os.path.join(experiment_Dir, 'full-data-config.yml'), 'w') as f:
            doc = yaml.dump(cfg, f)

        print('='*5, 'Model configs', '='*5)
        print(json.dumps(modelConfig, indent=1, sort_keys=True))
        print('='*2, 'Training configs', '='*5)
        print(json.dumps(trainConfig, indent=1, sort_keys=True))

    return modelConfig, trainConfig


if __name__ == '__main__':
    world_size = 1
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)


