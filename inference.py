from model_ead import TransformerXL
import pickle
import os
import torch
import yaml
import json
import make_loops as loops
import guitarpro

import numpy as np
import datetime
from dadagp import dadagp_decode, tokens2guitarpro

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
    return inference_config['gpuID']

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

def decode_outputs(experiment_dir):
    for item in os.listdir(experiment_dir):
        full_path = os.path.join(experiment_dir, item)
        if full_path.endswith(".txt"):
            print("Decoding {}...".format(item))
            output_path = os.path.join(experiment_dir, item[:-4] + ".gp5")
            dadagp_decode(full_path, output_path)

def extract_loops(experiment_dir):
    for idx, item in enumerate(os.listdir(experiment_dir)):
        full_path = os.path.join(experiment_dir, item)
        if full_path.endswith(".txt"):
            print("Extracting loops from {}...".format(item))
            text_file = open(full_path, "r")
            tokens = text_file.read().split("\n")
            song = tokens2guitarpro(tokens, verbose=False)
            extract_loops_from_song(song, tokens, idx, experiment_dir)

#calculate average number of notes per instrument in each measure
def calc_density(token_list):
    num_meas = 0
    timestamp = 0
    num_notes = {}
    for i in range(len(token_list)):
        t = token_list[i]
        if "note" in t:
            instrument = t.split(":")[0]
            if instrument not in num_notes:
                num_notes[instrument] = 1
            else:
                num_notes[instrument] += 1
        if t == "new_measure":
            num_meas += 1

    total_notes = 0
    for inst in num_notes.keys():
        total_notes += num_notes[inst]
    curr_density = total_notes * 1.0 / len(num_notes)

    return curr_density / num_meas

def extract_loops_from_song(song, generated_tokens, idx, output_dir):
    #loop extraction parameters
    LOOP_SIZE = 4
    MIN_LEN = 4
    MIN_REP_BEATS = 2.0
    DENSITY = 1

    #extract loops
    track_list, time_signatures = loops.create_track_list(song)
    beats_per_bar = 4 #inference forces 4/4
    min_beats = beats_per_bar * LOOP_SIZE
    max_beats = beats_per_bar * LOOP_SIZE
    lead_mat, lead_dur, melody_seq = loops.calc_correlation(track_list, 0) 
    _, loop_endpoints = loops.get_valid_loops(melody_seq, lead_mat, lead_dur, min_len=MIN_LEN, min_beats=min_beats, max_beats=max_beats, min_rep_beats=MIN_REP_BEATS)
    token_list = loops.unify_loops(generated_tokens, loop_endpoints, density=DENSITY)
    token_list_repeats = loops.get_repeats(generated_tokens, min_meas=LOOP_SIZE, max_meas=LOOP_SIZE, density=DENSITY)
    token_list = token_list + token_list_repeats
    if token_list[-1] != "end":
        token_list.append("end")

    loops_length = len(token_list)
    if loops_length > 10:
        header_data = token_list[:4]
        main_data = token_list[4:]

        num_measures = 0
        split_loops = []
        current_loop = []
        for i,token in enumerate(main_data):
            if token == "new_measure":
                if num_measures > 0 and num_measures % LOOP_SIZE == 0: #end of a loop
                    split_loops.append(current_loop)
                    current_loop = []
                num_measures += 1
            if token == "end":
                split_loops.append(current_loop)
                break
            current_loop.append(token)

        token_list = header_data

        num_segments = 0
        for i,loop in enumerate(split_loops):
            duplicate = False
            current = " ". join(split_loops[i])
            for j in range(0,i):
                comparison = " ".join(split_loops[j])
                if comparison == current:
                    duplicate = True
                    break
            if not duplicate:
                token_list += loop
                num_segments += 1
        token_list.append("end")

        density = calc_density(token_list)
        print("FOUND {} loops in ex_{}, density {}".format(num_segments, idx, density))

        #save extracted loops
        song = tokens2guitarpro(token_list, verbose=False)
        song.artist = generated_tokens[0]
        song.album = 'Generated by DadaGP'
        song.title = "untitled"
        dadagp_path = os.path.join(output_dir, "ex_" + str(idx) + "_loops" + ".gp5")
        guitarpro.write(song, dadagp_path)

        total_segments += num_segments


def run_inference(output_loc):
    model_config = yaml.full_load(open("model_config.yaml", 'r')) 
    inference_config = yaml.full_load(open("inference_config.yaml", 'r')) 

    device = get_device(inference_config)
    model = load_model(model_config, inference_config, device)
    model_path = get_model_path(inference_config)

    experiment_dir = create_output_dir(inference_config, output_loc)


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

    return experiment_dir

if __name__ == '__main__':
    output_loc = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_inference(output_loc)