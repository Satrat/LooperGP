'''
save_loops.py

Sara Adkins 2022
'''

import json
import os
import tqdm
import make_loops as loops
import dadagp as dada

# path to DadaGP dataset and file list
root_path = "D:\Documents\DATA\DadaGP-v1.1"
save_path = "D:\Documents\DATA\DadaGP-4-8-lps-3-dens-per-inst-hard-reps"
allfiles_path = os.path.join(root_path,"_DadaGP_all_filenames.json" )

# loop filter parameters
MIN_LEN = 4
MIN_BARS = 4.0
MAX_BARS = 8.0
MIN_REP_BEATS = 3.0
DENSITY = 3.0

# extract loops from original DadaGP dataset based on loop filter parameters
# save results to save_path
def process(filtered_files):  
    file_list = []
    n_files = len(filtered_files)

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
        song = loops.convert_from_dadagp(list_words)
        track_list, time_signatures = loops.create_track_list(song)
        beats_per_bar = loops.get_dom_beats_per_bar(time_signatures)
        min_beats = beats_per_bar * MIN_BARS
        max_beats = beats_per_bar * MAX_BARS
        lead_mat, lead_dur, melody_seq = loops.calc_correlation(track_list, 0) #assuming first instrument is primary melody
        _, loop_endpoints = loops.get_valid_loops(melody_seq, lead_mat, lead_dur, min_len=MIN_LEN, min_beats=min_beats, max_beats=max_beats, min_rep_beats=MIN_REP_BEATS)
        if len(loop_endpoints) > 0:
            loop_list = loops.unify_loops(list_words, loop_endpoints, density=DENSITY)
            loop_list_repeats = loops.get_repeats(list_words, min_meas=MIN_BARS,max_meas=MAX_BARS,density=DENSITY)
            loop_list = loop_list + loop_list_repeats
            header = list_words[0:4]
            footer = ["end"]
            if len(loop_list) > 0: #if loops found in song, save to loop dataset
                for idx, loop in enumerate(loop_list):
                    loop = header + loop + footer
                    if not os.path.exists(os.path.join(save_path, folder_name)):
                        os.makedirs(os.path.join(save_path, folder_name))
                    token_path = os.path.join(save_path, file_prefix + "_" + str(idx) + "_loops" + ".txt")
                    dadagp_path = os.path.join(save_path, file_prefix + "_" + str(idx) + "_loops" + ".gp5")
                    f = open(token_path, "w")
                    f.write("\n".join(loop))
                    f.close()
                    dada.dadagp_decode(token_path, dadagp_path)
                    file_list.append(file_prefix + "_" + str(idx) + "_loops" + ".txt")

    path_json = os.path.join(save_path, "file_list_loops.json")
    with open(path_json, 'w') as f:
        json.dump(file_list, f)

if __name__ == '__main__':

    with open(allfiles_path, "r") as f:
        allfiles =  json.load(f)

    process(allfiles, "None")