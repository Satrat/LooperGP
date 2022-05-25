from model_ead import TransformerXL
import pickle
import random
import os
import time
import torch
import random
import yaml
import json

import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main():
    config_file = "full-data-config_5_lat1024.yml"
    cfg = yaml.full_load(open("full-data-config_5_lat1024.yml", 'r')) 
    #print(cfg)
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
        output_prefix = 'best_train_'
    elif checkpoint_type == 'best_val':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best_val.pth.tar')
        output_prefix = 'best_val_'
    elif checkpoint_type == 'epoch_idx':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'ep_{}.pth.tar'.format(str(inferenceConfig['model_epoch'])))
        output_prefix = str(inferenceConfig['model_epoch'])+ '_'

    # Insert folder path for pre-trained model
    pretrainCfg = yaml.full_load(open(os.path.join('./model-weights',"full-data-config.yml"), 'r')) 
    modelConfig = pretrainCfg['MODEL']
    print(modelConfig['n_token'])

    # create result folder
    if not os.path.exists(midi_folder):
        os.mkdir(midi_folder)

    # load dictionary
    #event2word, word2event = pickle.load(open(inferenceConfig['dictionary_path'], 'rb'))
    event2word = pickle.load(open("vocab_song_artist.pkl", 'rb'))
    word2event = pickle.load(open("rev_vocab_song_artist.pkl", 'rb'))
    #print(event2word)

    # declare model
    device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)

    # declare model
    model =  TransformerXL(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=False)

    # inference
    song_time_list = []
    words_len_list = []
    num_samples = inferenceConfig["num_sample"]
    primer = 2      # use the empty prompt
    print(primer)
    # primer 4 -> E
    # primer 5 -> A
    # primer 6 -> D
    #tempi = np.arange(40,240,10)
    #bpm = random.choice(tempi)
    bpm = 100        # bpm initial
    songs_per_tonic = 0
    songs_per_bpm = 0
    for idx in range(num_samples):
        songs_per_tonic += 1
        songs_per_bpm += 1
        print(f'==={idx}/{num_samples}===')
        #print(midi_folder, output_prefix + str(idx))
        song_time, word_len = model.inference(
            model_path = model_path,
            token_lim=2048,
            strategies=['temperature', 'nucleus'],
            params={'t': 1.2 ,'p': 0.9},
            bpm=bpm,
            primer=primer,
            id = idx, 
            output_path="inference-attempts/model_5lat1024_ep200_2048_2")
        if songs_per_bpm == 10:
            songs_per_bpm = 0
            bpm += 20
        if songs_per_tonic == 50:
            primer += 1
            bpm = 80
            songs_per_tonic = 0
        

        # 50 songs in E
        #   10 at 80 bpm   
        #   10 at 100 bpm
        #   10 at 120 bpm
        #   10 at 140 bpm
        #   10 at 160 bpm
            
        print('song time:',  song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)
    

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