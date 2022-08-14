'''
inference.py

Pedro Sarmento, Adarsh Kumar, C J Carr, Zack Zukowski, Mathieu
Barthet, and Yi-Hsuan Yang. Dadagp: A dataset of tokenized guitarpro
songs for sequence models, 2021.
'''

from model_ead import TransformerXL
import pickle
import os
import torch
import yaml
import json

import numpy as np
import datetime

def main():
    cfg = yaml.full_load(open("full-data-config_5_lat1024.yml", 'r')) 
    inferenceConfig = cfg['INFERENCE']
    
    os.environ['CUDA_VISIBLE_DEVICES'] = inferenceConfig['gpuID']

    print('='*2, 'Inferenc configs', '='*5)
    print(json.dumps(inferenceConfig, indent=1, sort_keys=True))
    

    # checkpoint information
    CHECKPOINT_FOLDER = inferenceConfig['experiment_dir']
    midi_folder = inferenceConfig["generated_dir"]

    checkpoint_type = inferenceConfig['checkpoint_type']
    if checkpoint_type == 'best_train':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best.pth.tar')
    elif checkpoint_type == 'best_val':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best_val.pth.tar')
    elif checkpoint_type == 'epoch_idx':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'ep_{}.pth.tar'.format(str(inferenceConfig['model_epoch'])))

    # Insert folder path for pre-trained model
    pretrainCfg = yaml.full_load(open(os.path.join(CHECKPOINT_FOLDER,"full-data-config.yml"), 'r')) 
    modelConfig = pretrainCfg['MODEL']

    # create result folder
    if not os.path.exists(midi_folder):
        os.mkdir(midi_folder)

    # load dictionary
    event2word = pickle.load(open(inferenceConfig['vocab_data_path'], 'rb'))
    word2event = pickle.load(open(inferenceConfig['rev_vocab_data_path'], 'rb'))

    # declare model
    device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)

    # declare model
    model =  TransformerXL(
            modelConfig,
            inferenceConfig['gpuID'],
            event2word=event2word, 
            word2event=word2event, 
            is_training=False)

    # inference
    song_time_list = []
    words_len_list = []
    num_samples = inferenceConfig["num_sample"]
    bpm = inferenceConfig["bpm"]
    num_bars = inferenceConfig["num_bars"]
    key = inferenceConfig["key"]
    initial_wait = inferenceConfig["initial_wait"]    

    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_dir = os.path.join(inferenceConfig['generated_dir'],cur_date)
    if not os.path.exists(experiment_dir):
        print('Creating experiment_dir:', experiment_dir)
        os.makedirs(experiment_dir) 

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
    

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)

if __name__ == '__main__':
    main()