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
from primers import build_primer
from dataclasses import dataclass

@dataclass
class LoopExtractConfig:
    loop_size = 4
    min_length = 4
    min_rep_beats = 2.0
    density = 1.0

def load_model(model_config, inference_config, device):
    # load dictionary
    event2word = pickle.load(open(inference_config['vocab_data_path'], 'rb'))
    word2event = pickle.load(open(inference_config['rev_vocab_data_path'], 'rb'))
    model_path = get_model_path(inference_config)

    # declare model
    model =  TransformerXL(
            model_config,
            model_path,
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
        os.mkdir(experiment_dir) 
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

def extract_loops_from_song(song, generated_tokens, name, output_dir, loop_config):
    #loop extraction parameters
    LOOP_SIZE = loop_config.loop_size
    MIN_LEN = loop_config.min_len
    MIN_REP_BEATS = loop_config.min_rep_beats
    DENSITY = loop_config.density

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
        print("FOUND {} loops in {}, density {}".format(num_segments, name, density))

        #save extracted loops
        song = tokens2guitarpro(token_list, verbose=False)
        song.artist = generated_tokens[0]
        song.album = 'Generated by DadaGP'
        song.title = "untitled"
        dadagp_path = os.path.join(output_dir, str(name) + "_loops" + ".gp5")
        guitarpro.write(song, dadagp_path)

def run_single_inference(output_loc, name, model, num_bars, primer, loop_config):
    generated_tokens = model.inference_single_from_primer(['temperature', 'nucleus'], {'t': 1.2 ,'p': 0.9, 'num_bars': num_bars}, primer)

    #save raw generation as GuitarPro file
    song = tokens2guitarpro(generated_tokens, verbose=False)
    song.artist = generated_tokens[0]
    song.album = 'Generated by DadaGP'
    song.title = "untitled"
    dadagp_path = os.path.join(output_loc, name + "_full" + ".gp5")
    guitarpro.write(song, dadagp_path)
    extract_loops_from_song(song, generated_tokens, name, output_loc, loop_config)

def setup_inference(output_loc):
    model_config = yaml.full_load(open("model_config.yaml", 'r')) 
    inference_config = yaml.full_load(open("inference_config.yaml", 'r')) 

    device = get_device(inference_config)
    model = load_model(model_config, inference_config, device)
    experiment_dir = create_output_dir(inference_config, output_loc)

    return experiment_dir, model

if __name__ == '__main__':
    output_loc = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_dir, model = setup_inference(output_loc)

    for idx in range(5):
        name = "example_" + str(idx)
        primer = build_primer(120, key="a", duration=240)
        bars_to_generate = 8
        loop_extract_config = LoopExtractConfig() #defaults
        run_single_inference(experiment_dir, name, model, bars_to_generate, primer, loop_extract_config)