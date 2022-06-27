import json
import pickle 
import numpy as np
import os
import random
import tqdm
import make_loops as loops
import copy
import dadagp as dada
import guitarpro

# PATHS
root_path = "D:\Documents\DATA\DadaGP-v1.1"
save_path = "D:\Documents\DATA\DadaGP-Loops"
allfiles_path = os.path.join(root_path,"_DadaGP_all_filenames.json" )

# for loops
MIN_LEN = 4
MIN_BEATS = 16.0
MAX_BEATS = 16.0
MIN_REP_BEATS = 2.0

# To turn dictionary into .npz files
file_list = []
def process(filtered_files, fname="", save_loops=True):  
    n_files = len(filtered_files)

    # process
    for fidx in tqdm.tqdm(range(n_files)):
        
        file = os.path.join(root_path, filtered_files[fidx])

        if save_loops:
            folder_name, file_name = os.path.split(filtered_files[fidx])
            file_name = file_name[:-4] #remove .txt
            file_prefix = folder_name + "/" + file_name

            try:
                with open(file, "r") as f:
                    text = f.read()
            except:
                print("ERROR: Skipping unreadable file {}".format(file))
                continue

            list_words = text.split("\n")
            song = loops.convert_from_dadagp(list_words)
            track_list = loops.create_track_list(song)
            lead_mat, lead_dur, melody_seq = loops.calc_correlation(track_list, 0) #assuming first instrument is most loopable
            _, loop_endpoints = loops.get_valid_loops(melody_seq, lead_mat, lead_dur, min_len=MIN_LEN, min_beats=MIN_BEATS, max_beats=MAX_BEATS, min_rep_beats=MIN_REP_BEATS)
            #print(loop_endpoints)
            for i, endpoints in enumerate(loop_endpoints):
                new_song = loops.convert_gp_loops(copy.deepcopy(song), endpoints)
                tokens = dada.guitarpro2tokens(new_song, new_song.artist, verbose=False)
                if not os.path.exists(os.path.join(save_path, folder_name)):
                    os.makedirs(os.path.join(save_path, folder_name))
                token_path = os.path.join(save_path, file_prefix + "_loop" + str(i) + ".txt")
                dadagp_path = os.path.join(save_path, file_prefix + "_loop" + str(i) + ".gp5")
                guitarpro.write(new_song, dadagp_path)
                dada.dadagp_encode(dadagp_path, token_path, song.artist)

    path_json = os.path.join(save_path, "file_list.json")
    with open(path_json, 'w') as f:
        json.dump(file_list, f)

if __name__ == '__main__':

    with open(allfiles_path, "r") as f:
        allfiles =  json.load(f)

    process(allfiles, "None")