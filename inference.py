from model_ead import TransformerXL
import pickle
import os
import torch
import yaml
import json

import numpy as np
import datetime

def load_model(model_config, inference_config, device):
    # load dictionary
    event2word = pickle.load(open(inference_config['vocab_data_path'], 'rb'))
    word2event = pickle.load(open(inference_config['rev_vocab_data_path'], 'rb'))

    # declare model
    model =  TransformerXL(
            model_config,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=False)
    return model
    
def get_device(inference_config):
    os.environ['CUDA_VISIBLE_DEVICES'] = inference_config['gpuID']
    device = torch.device("cuda" if not inference_config["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)
    return device

def get_model_path(inference_config):
    # checkpoint information
    CHECKPOINT_FOLDER = inference_config['experiment_dir']

    checkpoint_type = inference_config['checkpoint_type']
    if checkpoint_type == 'best_train':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best.pth.tar')
    elif checkpoint_type == 'best_val':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best_val.pth.tar')
    elif checkpoint_type == 'epoch_idx':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'ep_{}.pth.tar'.format(str(inference_config['model_epoch'])))

    return model_path

def create_output_dir(inference_config, output_name):
    parent_output_dir = inference_config['generated_dir']
    if not os.path.exists(parent_output_dir):
        os.mkdir(parent_output_dir)

    experiment_dir = os.path.join(inference_config['generated_dir'], output_name)
    if not os.path.exists(experiment_dir):
        print('Creating experiment_dir:', experiment_dir)
        os.makedir(experiment_dir) 
    return experiment_dir

def run_inference():
    model_config = yaml.full_load(open("model_config.yaml", 'r')) 
    inference_config = yaml.full_load(open("inference_config.yaml", 'r')) 

    device = get_device(inference_config)
    model = load_model(model_config, inference_config, device)
    model_path = get_model_path(inference_config)

    curr_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_dir = create_output_dir(inference_config, curr_date)


    # inference
    song_time_list = []
    words_len_list = []
    num_samples = inference_config["num_sample"]
    bpm = 120
    num_bars = 16
    key = "a"
    initial_wait = 240  

    for idx in range(num_samples):
        print(f'==={idx}/{num_samples}===')
        song_time, word_len = model.inference(
            model_path = model_path,
            strategies=['temperature', 'nucleus'],
            params={'t': 1.5 ,'p': 0.9, 'bpm': bpm, 'num_bars':num_bars, 'key':key, 'initial_wait':initial_wait},
            id = idx, 
            output_path=experiment_dir)
            
        print('song time:',  song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)
    

    print("=========")
    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time':song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

if __name__ == '__main__':
    run_inference()