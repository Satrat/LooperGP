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
save_path = "D:\Documents\DATA\DadaGP-4-8-lps-3-dens-per-inst-hard-reps"
allfiles_path = os.path.join(root_path,"_DadaGP_all_filenames.json" )

# for loops
MIN_LEN = 4
MIN_BARS = 4.0
MAX_BARS = 8.0
MIN_REP_BEATS = 3.0

# To turn dictionary into .npz files
file_list = []
def process(filtered_files, fname=""):  
    n_files = len(filtered_files)

    # process
    for fidx in tqdm.tqdm(range(n_files)):
        
        file = os.path.join(root_path, filtered_files[fidx])

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
        #print(len(list_words))
        song = loops.convert_from_dadagp(list_words)
        track_list, time_signatures = loops.create_track_list(song)
        beats_per_bar = loops.get_dom_beats_per_bar(time_signatures)
        min_beats = beats_per_bar * MIN_BARS
        max_beats = beats_per_bar * MAX_BARS
        lead_mat, lead_dur, melody_seq = loops.calc_correlation(track_list, 0) #assuming first instrument is most loopable
        _, loop_endpoints = loops.get_valid_loops(melody_seq, lead_mat, lead_dur, min_len=MIN_LEN, min_beats=min_beats, max_beats=max_beats, min_rep_beats=MIN_REP_BEATS)
        if len(loop_endpoints) > 0:
            print(file_prefix)
            loop_list = loops.unify_loops(list_words, loop_endpoints, density=3) #TODO: bad to hardcode loop len
            print(len(loop_list))
            #loop_list_repeats = loops.get_repeats(list_words, min_meas=MIN_BARS,max_meas=MAX_BARS,density=3)
            #print(len(loop_list_repeats))
            #loop_list = loop_list + loop_list_repeats
            header = list_words[0:4]
            footer = ["end"]
            if len(loop_list) > 0:
                for idx, loop in enumerate(loop_list):
                    loop = header + loop + footer
                    if not os.path.exists(os.path.join(save_path, folder_name)):
                        os.makedirs(os.path.join(save_path, folder_name))
                    token_path = os.path.join(save_path, file_prefix + "_" + str(idx) + "_loops" + ".txt")
                    dadagp_path = os.path.join(save_path, file_prefix + "_" + str(idx) + "_loops" + ".gp5")
                #print(len(token_path))
                    f = open(token_path, "w")
                    f.write("\n".join(loop))
                    f.close()
                    dada.dadagp_decode(token_path, dadagp_path)
                    file_list.append(file_prefix + "_" + str(idx) + "_loops" + ".txt")
        #for i, endpoints in enumerate(loop_endpoints):
        #    new_song = loops.convert_gp_loops(copy.deepcopy(song), endpoints)
        #    if new_song is not None:
        #        if not os.path.exists(os.path.join(save_path, folder_name)):
        #            os.makedirs(os.path.join(save_path, folder_name))
        #        token_path = os.path.join(save_path, file_prefix + "_loop" + str(i) + ".txt")
        #        dadagp_path = os.path.join(save_path, file_prefix + "_loop" + str(i) + ".gp5")
        #        guitarpro.write(new_song, dadagp_path)
        #        dada.dadagp_encode(dadagp_path, token_path, song.artist)
        #        file_list.append(file_prefix + "_loop" + str(i) + ".txt")

    path_json = os.path.join(save_path, "file_list_loops.json")
    with open(path_json, 'w') as f:
        json.dump(file_list, f)

if __name__ == '__main__':

    with open(allfiles_path, "r") as f:
        allfiles =  json.load(f)

    process(allfiles, "None")