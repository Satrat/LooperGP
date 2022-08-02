import json
import pickle 
import numpy as np
import os
import random
import tqdm
import make_loops as loops
import copy

# PATHS
root_path = "D:\Documents\DATA\DadaGP-4-8-lps-3-dens-per-inst-hard-reps" #"D:\Documents\DATA\DadaGP-Loops-many"
save_path = "D:\Documents\DATA\DadaGP-4-8-lps-3-dens-per-inst-hard-reps" #"D:\Documents\DATA\DadaGP-Output-many"
allfiles_path = os.path.join(root_path,"file_list_loops.json" ) 
# GLOBAL VARIABLES FOR PROCESS
WINDOW_SIZE = 512
GROUP_SIZE = 12  #15
MIN_LEN = 60
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = 'XL' # 'linear', 'XL'
VAL_SPLIT = 0.15
print('[config] MAX_LEN:', MAX_LEN)


# To turn dictionary into .npz files
def process(filtered_files, fname=""):  

    fname = "splitted"

    filtered_files = filtered_files

    event2word = pickle.load(open("../data/vocab_song_artist.pkl", 'rb'))
    eos_id = event2word['end']
    print(' > eos_id:', eos_id)

    # init
    x_list = []
    y_list = []
    mask_list = []
    seq_len_list = []
    num_groups_list = []
    name_list = []
    n_files = len(filtered_files)

    # process
    num_skipped = 0
    for fidx in tqdm.tqdm(range(n_files)):
        
        file = os.path.join(root_path, filtered_files[fidx])

        try:
            with open(file, "r") as f:
                text = f.read()
        except:
            print("ERROR: Skipping unreadable file {}".format(file))
            continue

        list_words = text.split("\n")
        words = []
        for i in list_words:
            try:
                words.append(event2word[i])
            except:
                print("ERROR: Skipping unrecognized token: {}".format(i))
        num_words = len(words)
        #print(num_words)

        if num_words >= MAX_LEN - 2: # 2 for room
            print(' [!] too long:', num_words)
            num_skipped += 1
            continue
        if num_words <=MIN_LEN:
            print(' [!] too short:', num_words)
            continue

        # arrange IO
        x = words[:-1]
        y = words[1:]     # Shifts!
        seq_len = len(x)
        #print(' > seq_len:', seq_len)

        # pad with eos
        x = np.concatenate([x, np.ones(MAX_LEN-seq_len) * eos_id])
        y = np.concatenate([y, np.ones(MAX_LEN-seq_len) * eos_id])
        mask = np.concatenate(
            [np.ones(seq_len), np.zeros(MAX_LEN-seq_len)])
        
        # collect
        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)
        seq_len_list.append(seq_len)
        num_groups_list.append(int(np.ceil(seq_len/WINDOW_SIZE)))
        name_list.append(file)
        #print(name_list)
        if fidx%1000==0:
            print("Collecting something", len(x_list))

    # sort by length (descending) 
    zipped = zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list)
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip( 
                            *sorted(zipped, key=lambda x: -x[0])) 

    print('\n\n[Finished]')
    print("SKIPPED ", num_skipped, " out of ", n_files)
    print(' compile target:', COMPILE_TARGET)

    if COMPILE_TARGET == 'XL':
        x_final = np.array(x_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        print("x_final complete")
        y_final = np.array(y_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        print("y_final complete")
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        print("mask_final complete")
    elif COMPILE_TARGET == 'linear':
        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError('Unknown target:', COMPILE_TARGET)
        
    num_samples = len(seq_len_list)
    print(' >   count:', num_samples)
    print(' >   discarded: ',n_files-num_samples)
    print(' > x_final:', x_final.shape)
    print(' > y_final:', y_final.shape)
    print(' > mask_final:', mask_final.shape)

    # split train/test
    size = int(VAL_SPLIT*len(name_list))

    validation_songs = random.sample(name_list, size)

    # validation filename map
    fn_idx_map = {
        'fn2idx': dict(),
        'idx2fn': dict(),
    }

    train_idx = []
    test_idx = []

    # run split
    valid_cnt = 0
    for nidx, n in enumerate(name_list):
        flag = True
        for fn in validation_songs:  
            if fn in n:
                test_idx.append(nidx)
                flag = False
                fn_idx_map['fn2idx'][fn] = valid_cnt
                fn_idx_map['idx2fn'][valid_cnt] = fn
                valid_cnt += 1
                break
        if flag:
            train_idx.append(nidx)  
    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)

    # save validation map 
    path_json = os.path.join(save_path, fname + '_valid_fn_idx_map.json')
    with open(path_json, 'w') as f:
        json.dump(fn_idx_map, f)

    # save train
    path_train = os.path.join(save_path, '{}_train_data_{}.npz'.format(fname, COMPILE_TARGET))
    np.savez(
        path_train, 
        x=x_final[train_idx],
        y=y_final[train_idx],
        mask=mask_final[train_idx],
        seq_len=np.array(seq_len_list)[train_idx],        
        num_groups=np.array(num_groups_list)[train_idx]
    )

    # save test
    path_test = os.path.join(save_path, '{}_test_data_{}.npz'.format(fname, COMPILE_TARGET))
    np.savez(
        path_test, 
        x=x_final[test_idx],
        y=y_final[test_idx],
        mask=mask_final[test_idx],
        seq_len=np.array(seq_len_list)[test_idx],
        num_groups=np.array(num_groups_list)[test_idx]
    )

    print('---')
    print(' > train x:', x_final[train_idx].shape)
    print(' >  test x:', x_final[test_idx].shape)                       


if __name__ == '__main__':

    with open(allfiles_path, "r") as f:
        allfiles =  json.load(f)

    print(len(allfiles))
    process(allfiles, "None")