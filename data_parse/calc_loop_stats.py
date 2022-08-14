'''
calc_loop_stats.py

Sara Adkins 2022
'''

import json
import os
import tqdm
import make_loops as loops

# path to DadaGP loop dataset and file list
root_path = "D:\Documents\DATA\DadaGP-v1.1"
allfiles_path = os.path.join(root_path,"_DadaGP_all_filenames.json" )

# loop filter parameters
MIN_BARS = 4.0
MAX_BARS = 16.0
DENSITY = 4.0
MIN_LEN = 4
MIN_REP_BEATS = 4.0

print("Num Bars: {} - {}".format(MIN_BARS, MAX_BARS))
print("Density: {}". format(DENSITY))
print("Min Rep Notes: {}, Min Rep Beats: {}". format(MIN_LEN, MIN_REP_BEATS))

#calculate the number of loops in the dataset under the selected filter parameters
#does NOT save the loops to file
def process(filtered_files):  
    num_ignored = 0
    n_files = len(filtered_files)

    total_loops = 0
    for fidx in tqdm.tqdm(range(n_files)):
        
        file = os.path.join(root_path, filtered_files[fidx])

        _, file_name = os.path.split(filtered_files[fidx])
        file_name = file_name[:-4] #remove .txt

        try:
            with open(file, "r") as f:
                text = f.read()
        except:
            print("ERROR: Skipping unreadable file {}".format(file))
            num_ignored += 1
            continue

        list_words = text.split("\n")
        song = loops.convert_from_dadagp(list_words)
        track_list, time_signatures = loops.create_track_list(song)
        beats_per_bar = loops.get_dom_beats_per_bar(time_signatures)
        min_beats = beats_per_bar * MIN_BARS
        max_beats = beats_per_bar * MAX_BARS
        lead_mat, lead_dur, melody_seq = loops.calc_correlation(track_list, 0) #assuming first instrument is primary melody
        _, loop_endpoints = loops.get_valid_loops(melody_seq, lead_mat, lead_dur, min_len=MIN_LEN, min_beats=min_beats, max_beats=max_beats, min_rep_beats=MIN_REP_BEATS)
        loop_endpoints = loops.filter_loops_density(list_words, loop_endpoints, density=DENSITY)
        num_repeats = loops.get_num_repeats(list_words, min_meas=MIN_BARS,max_meas=MAX_BARS,density=DENSITY)
        total_loops += len(loop_endpoints) + num_repeats

    print(n_files, num_ignored)
    n_files = n_files - num_ignored
    print("{} Loops in {} Files, {} Average Loops per File".format(total_loops, n_files, 1.0 * total_loops / n_files))

if __name__ == '__main__':

    with open(allfiles_path, "r") as f:
        allfiles =  json.load(f)

    process(allfiles, "None")