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
save_path = "D:\Documents\DATA\DadaGP-Loops-many"
allfiles_path = os.path.join(root_path,"_DadaGP_all_filenames.json" )

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
        print(file_prefix)

        try:
            with open(file, "r") as f:
                text = f.read()
        except:
            print("ERROR: Skipping unreadable file {}".format(file))
            continue

        list_words = text.split("\n")
        num_words = len(list_words)
        endpoint_dict = {}
        length_dict = {}
        open_reps = []
        curr_length = 0
        curr_notes = 0
        for i in range(num_words-2):
            t = list_words[i]
            if "note" in t:
                curr_notes += 1
            if t == "new_measure":
                curr_length += 1
                if list_words[i+1] == "measure:repeat_open":
                    curr_length = 1
                    curr_notes = 0
                    open_reps.append(i)
                    endpoint_dict[i] = -1
                if "measure:repeat_close" in list_words[i+1] or "measure:repeat_close" in list_words[i+2]:
                    if len(open_reps) > 0:
                        idx = open_reps.pop(len(open_reps) - 1)
                        endpoint_dict[idx] = i
                        length_dict[idx] = (curr_length, curr_notes)
                    elif len(endpoint_dict) == 0:
                        endpoint_dict[0] = i
                        length_dict[0] = (curr_length, curr_notes)

        if len(endpoint_dict) > 0:
            final_list = list_words[0:4]
            for start in endpoint_dict.keys():
                end = endpoint_dict[start]
                if end <= 0:
                    continue
                length_meas = length_dict[start][0]
                length_notes = length_dict[start][1]
                if length_meas < 4 or length_meas > 16 or length_notes < 4 * length_meas:
                    continue

                end += 1
                while(end < num_words and end >= 0):
                    if list_words[end] == "new_measure":
                        break
                    end += 1
                if end > start:
                    final_list += list_words[start:end]

            if len(final_list) > 50:
                if final_list[len(final_list) - 1] != "end":
                    final_list.append("end")

                if not os.path.exists(os.path.join(save_path, folder_name)):
                    os.makedirs(os.path.join(save_path, folder_name))
                token_path = os.path.join(save_path, file_prefix + "_repeats" + ".txt")
                dadagp_path = os.path.join(save_path, file_prefix + "_repeats" + ".gp5")
                f = open(token_path, "w")
                f.write("\n".join(final_list))
                f.close()
                dada.dadagp_decode(token_path, dadagp_path)
                file_list.append(file_prefix + "_repeats" + ".txt")

    path_json = os.path.join(save_path, "file_list2.json")
    with open(path_json, 'w') as f:
        json.dump(file_list, f)

if __name__ == '__main__':

    with open(allfiles_path, "r") as f:
        allfiles =  json.load(f)

    process(allfiles, "None")