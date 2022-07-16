from distutils.command.build import build
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import miditoolkit
import os
import time
from sklearn.model_selection import train_test_split
from modules import MemTransformerLM
from glob import glob
from transformers import AdamW, WarmUp, get_polynomial_decay_schedule_with_warmup

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import numpy as np

import saver
from torch.nn.parallel import DistributedDataParallel as DDP
from primers import build_primer

# Constants #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0, 'melody': 1}


def write_midi(words, path_midi, word2event):
    notes_all = []

    events = [word2event[words[i]] for i in range(len(words))]

    bar_cnt = 0
    cur_beat = 0

    midi_obj = miditoolkit.midi.parser.MidiFile()
    cur_pos = 0
    
    for i in range(len(events)-3):
        cur_event = events[i]
        name = cur_event.split('_')[0]
        attr = cur_event.split('_')
        if name == 'Bar':
            bar_cnt += 1
        elif name == 'Beat':
            cur_beat = int(attr[1])
            cur_pos = bar_cnt * BAR_RESOL + cur_beat * TICK_RESOL
        elif name == 'Chord':
            chord_text = attr[1] + '_' + attr[2]
            midi_obj.markers.append(Marker(text=chord_text, time=cur_pos))
        elif name == 'Tempo':
            midi_obj.tempo_changes.append(
                TempoChange(tempo=int(attr[1]), time=cur_pos))
        else:
            if 'Note_Pitch' in events[i] and \
            'Note_Velocity' in events[i+1] and \
            'Note_Duration' in events[i+2]:

                pitch = int(events[i].split('_')[-1])
                duration = int(events[i+2].split('_')[-1])

                if int(duration) == 0:
                    duration = 60

                end = cur_pos + duration 
                velocity = int(events[i+1].split('_')[-1])
                notes_all.append(
                    Note(pitch=pitch, start=cur_pos, end=end, velocity=velocity))
                
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_midi)


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class TransformerXL(object):
    def __init__(self, modelConfig, device_rank, event2word, word2event, is_training=True):

        self.event2word = event2word
        self.word2event = word2event
        self.modelConfig = modelConfig

        # model settings    
        self.n_layer= modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.seq_len= modelConfig['seq_len']
        self.mem_len =  modelConfig['mem_len']

        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.eval_tgt_len = modelConfig['eval_tgt_len']

        self.init = modelConfig['init']
        self.init_range = modelConfig['init_range']
        self.init_std = modelConfig['init_std']
        self.proj_init_std = modelConfig['proj_init_std']

        #mode
        self.is_training = is_training
        self.rank = device_rank
        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda:" + str(self.rank)
        print(self.device)
        

    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)
            
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)


    def get_model(self, pretrain_model=None):
        model = MemTransformerLM(self.modelConfig, is_training=self.is_training)

        st_eopch = 0
        if pretrain_model:
            checkpoint = torch.load(pretrain_model, map_location=self.device)
            print('Pretrained model config for {}: epoch {} best_loss {}'.format(self.device, checkpoint['epoch'], checkpoint['best_loss']))
            model.load_state_dict(checkpoint['state_dict'])
            print('{} loaded on {}'.format(pretrain_model, self.device))  
            st_eopch = checkpoint['epoch'] + 1

        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init) 
        return st_eopch ,model.to(self.device)


    def save_checkpoint(self, state, root, save_freq=10):
        if state['epoch'] % save_freq == 0:
            torch.save(state, os.path.join(root,'ep_{}.pth.tar'.format(state['epoch'])))

    def train_loss_record(self, epoch, train_loss,checkpoint_dir, val_loss=None):
        if val_loss:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss],
                    'val_loss': ['%.3f'%val_loss]})
            
        else:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss]})

        csv_file = os.path.join(checkpoint_dir, 'loss.csv')

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(os.path.join(checkpoint_dir, 'loss.csv'), mode='a', header=False,  index=False)

    def train(self, train_data, trainConfig, resume):
        if self.rank == 0:
            checkpoint_dir = trainConfig['experiment_dir']
            saver_agent = saver.Saver(checkpoint_dir)
        else:
            checkpoint_dir = None
            saver_agent = None

        batch_size = trainConfig['batch_size']
        torch.manual_seed(trainConfig["seed"])

        #Prepare model
        if resume != 'None':
            st_epoch, model = self.get_model(resume)
            print('Continue to train from {} epoch on {}'.format(st_epoch, self.device))
        else:
            st_epoch, model = self.get_model()

        model = DDP(model, device_ids=[self.rank])
        optimizer = optim.Adam(model.parameters(), lr=trainConfig['lr'])
       
        epoch_train_loss = []
        save_freq = trainConfig['save_freq']
        
        n_parameters = network_paras(model)
        print('n_parameters: {:,} on {}'.format(n_parameters, self.device))
        if self.rank == 0:
            saver_agent.add_summary_msg(
                ' > params amount: {:,d}'.format(n_parameters))

        # unpack
        train_x = train_data['x'] 
        train_y = train_data['y'] 
        mask = train_data['mask'] 
        num_groups = train_data['num_groups'] 

        num_batches = len(train_x ) // batch_size
        print("{} batches, {} per batch on {}".format(num_batches, batch_size, self.device))
        
        print('>>> Start training on {}'.format(self.device))
        for epoch in range(st_epoch, trainConfig['num_epochs']):
            if self.rank == 0:
                saver_agent.global_step_increment()

            train_loss = []
            st_time = time.time()
            model.train()

            for bidx in range(num_batches):
                
                model.zero_grad()

                # index
                bidx_st = batch_size * bidx
                bidx_ed = batch_size * (bidx + 1)

                # get batch
                batch_x = train_x[bidx_st:bidx_ed]
                batch_y = train_y[bidx_st:bidx_ed]
                batch_mask = mask[bidx_st:bidx_ed]
                n_group  = np.max(num_groups[bidx_st:bidx_ed])

                # proc groups
                mems = tuple()
                for gidx in range(n_group):
                    group_x = batch_x[:, gidx, :]
                    group_y = batch_y[:, gidx, :]
                    group_mask = batch_mask[:, gidx, :]
                    
                    group_x = torch.from_numpy(group_x).permute(1, 0).contiguous().to(self.device).long()  # (seq_len, bsz)
                    group_y = torch.from_numpy(group_y).permute(1, 0).contiguous().to(self.device).long()
                    group_mask = torch.from_numpy(group_mask).to(self.device).float()
                    
                    ret = model(group_x, group_y, group_mask, *mems)
                    loss, mems = ret[0], ret[1:]              
                    train_loss.append(loss.item()) 
                    loss.backward()

                    if self.rank == 0:
                        sys.stdout.write('epoch:{:3d}/{:3d}, batch: {:4d}/{:4d}, group: {:2d}/{:2d} | Loss: {:6f}\r'.format(
                            epoch,
                            trainConfig['num_epochs'],
                            bidx,
                            num_batches,
                            gidx,
                            n_group, 
                            loss.item()
                        ))
                        sys.stdout.flush()

                optimizer.step()

            if self.rank == 0:
                curr_train_loss = sum(train_loss) / len(train_loss)
                saver_agent.add_summary('epoch loss', curr_train_loss)

                epoch_train_loss.append(curr_train_loss)
                epoch_info = 'Epoch: {}, Train Loss: {:.5f} ,  T: {:.3f}'.format(epoch+1, curr_train_loss, time.time()-st_time)
                print(epoch_info)

                self.train_loss_record(epoch, curr_train_loss, checkpoint_dir)
                self.save_checkpoint({
                        'epoch': epoch + 1,
                        'model_setting': self.modelConfig,
                        'train_setting': trainConfig,
                        'state_dict': model.state_dict(),
                        'best_loss': curr_train_loss,
                        'optimizer' : optimizer.state_dict(),
                                    }, 
                        checkpoint_dir, 
                        save_freq)

                if curr_train_loss < 0.01:
                    print('Experiment [{}] finished at loss < 0.01.'.format(checkpoint_dir))
                    break

    def inference(self, model_path, strategies, params, id, output_path):

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        _, model = self.get_model(model_path)
        model.eval()
        
        # initial start
        words = [[]]

        # text string to save into file
        final = str()

        bpm = params['bpm']
        key = params['key']
        initial_wait = params['initial_wait']     
        ticks_since_measure = 0
        beg_list = build_primer(bpm, key=key, duration=initial_wait)
        ticks_since_measure += initial_wait

        # FOR THE SPLITTED APPROACH
        others_splitted=[]
        for i in beg_list:
            token = i
            if ":" in token:
                token = token.split(":")
                h_tokens=[]
                for i in token[:-1]:
                    h_tokens.append(i+":")
                h_tokens.append(token[-1])
                others_splitted+=h_tokens
            else: 
                others_splitted.append(token)
       
        print("Primer: {}".format(beg_list))


        beg_event2word = list()
        for ele in beg_list:
            beg_event2word.append(self.event2word[ele])

        words[-1] += beg_event2word #[[2982, 476, 974, 440, 4, 19, 36, 2, 5, 0]]
        final = "\n".join(beg_list) 
        final+='\n'

        # initialize mem
        mems = tuple()
        song_init_time = time.time()

        # generate
        initial_flag = True
        generate_n_bar = 0 #since were priming with 0
        batch_size = 1
        n_tokens = len(words[0])
        ticks_per_measure = 960 * 4
        bars_to_generate = params['num_bars']
        measures_since_repeat = 1
        while generate_n_bar < bars_to_generate:
            # prepare input
            if initial_flag:
                temp_x = np.zeros((len(words[0]), batch_size))

                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[z][b] = t
                initial_flag = False
            else:
                temp_x = np.zeros((1, batch_size))
                
                for b in range(batch_size):
                    temp_x[0][b] = words[b][-1] ####?####

            temp_x = torch.from_numpy(temp_x).long().to(self.device)     
            
            _logits, mems = model.generate(temp_x, *mems) # logits is the probability of each token
            logits = _logits.cpu().squeeze().detach().numpy()

            # temperature or not
            if 'temperature' in strategies:
                probs = self.temperature(logits=logits, temperature=params['t'])
                
            else:
                probs = self.temperature(logits=logits, temperature=1.)

            word = self.nucleus(probs=probs, p=params['p']) 
            #print(word, self.word2event[word])

            #CONDITION TO TACKLE THE "measure:repeat"
            flag=0
            while flag==0:
                if  "measure" in self.word2event[word] and "repeat" in self.word2event[word] and len(self.word2event[word].split(":"))>2 and int(self.word2event[word].split(":")[-1])>4:
                    probs = self.temperature(logits=logits, temperature=10) #10
                    word = self.nucleus(probs=probs, p=0.9) #0.99
                else:
                   flag=1

            # CONDITION TO TACKLE THE "artist:"
            # which seems to pop up if the prompt is empty (?)
            if 'artist' in self.word2event[word]:
                probs = self.temperature(logits=logits, temperature=10) #10
                word = self.nucleus(probs=probs, p=0.99) #0.99

            # CONDITION TO TACKLE THE "end"
            if self.word2event[word] == 'end':
                probs = self.temperature(logits=logits, temperature=10) #5
                word = self.nucleus(probs=probs, p=0.99) #0.99

            # take repeats into account when counting measures
            skip_token = False
            if "measure" in self.word2event[word] and "repeat" in self.word2event[word]:
                split_word = self.word2event[word].split(":")
                if "open" in self.word2event[word]:
                    measures_since_repeat = 1
                if "close" in self.word2event[word]:
                    num_repeats = int(split_word[2])
                    total_measures = num_repeats * measures_since_repeat
                    if generate_n_bar + total_measures > bars_to_generate:
                        skip_token = True
                    elif measures_since_repeat == 0:
                        skip_token = True
                    else:
                        generate_n_bar += total_measures
                    measures_since_repeat = 0

            #enforce time signature
            new_measure = False
            if 'wait' in self.word2event[word]:
                split_word = self.word2event[word].split(":")
                wait_amnt = int(split_word[1])
                if ticks_since_measure + wait_amnt > ticks_per_measure:
                    print(ticks_per_measure, ticks_since_measure)
                    new_wait_amnt = ticks_per_measure - ticks_since_measure
                    word = self.event2word["wait:" + str(new_wait_amnt)]
                    new_measure = True
                    ticks_since_measure = 0
                elif ticks_since_measure + wait_amnt == ticks_per_measure:
                    new_measure = True
                    ticks_since_measure = 0
                else:
                    ticks_since_measure += wait_amnt
            elif "new_measure" == self.word2event[word]:
                skip_token = True
            elif "tempo" in self.word2event[word]:
                skip_token = True
            
            if not skip_token:
                words[0].append(word)
                final += self.word2event[word] + '\n'
            if new_measure:
                event = 'new_measure'
                words[0].append(self.event2word[event])
                final += event + '\n'
                generate_n_bar += 1
                measures_since_repeat += 1


        temperatura = params['t']
        parametro_n = params['p']

        generated_file_name = output_path + "/gentokens_" + "t_" + str(temperatura) + "_p_"+ str(parametro_n) + "_id_"+ str(id) +".txt"

        with open(generated_file_name, "w") as text_file:
            final += 'end\n'
            text_file.write(final)
        #write_midi(words[0], output_path, self.word2event)       
        

        song_total_time = time.time() - song_init_time
        return song_total_time, len(words[0])

    def inference_single_from_primer(self, model_path, strategies, params, primer):
        _, model = self.get_model(model_path)
        model.eval()
        
        # initial start
        words = [[]]

        # text string to save into file
        final = str()
    
        ticks_since_measure = 0
        beg_list = primer

        # FOR THE SPLITTED APPROACH
        others_splitted=[]
        for i in beg_list:
            token = i
            if ":" in token:
                token = token.split(":")
                h_tokens=[]
                for i in token[:-1]:
                    h_tokens.append(i+":")
                h_tokens.append(token[-1])
                others_splitted+=h_tokens
            else: 
                others_splitted.append(token)
        
        print("Primer: {}".format(beg_list))


        beg_event2word = list()
        for ele in beg_list:
            beg_event2word.append(self.event2word[ele])

        words[-1] += beg_event2word #[[2982, 476, 974, 440, 4, 19, 36, 2, 5, 0]]
        final = "\n".join(beg_list) 
        final+='\n'

        # initialize mem
        mems = tuple()
        song_init_time = time.time()

        # generate
        initial_flag = True
        generate_n_bar = 0 #since were priming with 0
        batch_size = 1
        n_tokens = len(words[0])
        ticks_per_measure = 960 * 4
        bars_to_generate = params['num_bars']
        measures_since_repeat = 1
        while generate_n_bar < bars_to_generate:
            # prepare input
            if initial_flag:
                temp_x = np.zeros((len(words[0]), batch_size))

                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[z][b] = t
                initial_flag = False
            else:
                temp_x = np.zeros((1, batch_size))
                
                for b in range(batch_size):
                    temp_x[0][b] = words[b][-1] ####?####

            temp_x = torch.from_numpy(temp_x).long().to(self.device)     
            
            _logits, mems = model.generate(temp_x, *mems) # logits is the probability of each token
            logits = _logits.cpu().squeeze().detach().numpy()

            # temperature or not
            if 'temperature' in strategies:
                probs = self.temperature(logits=logits, temperature=params['t'])
                
            else:
                probs = self.temperature(logits=logits, temperature=1.)

            word = self.nucleus(probs=probs, p=params['p']) 
            #print(word, self.word2event[word])

            #CONDITION TO TACKLE THE "measure:repeat"
            flag=0
            while flag==0:
                if  "measure" in self.word2event[word] and "repeat" in self.word2event[word] and len(self.word2event[word].split(":"))>2 and int(self.word2event[word].split(":")[-1])>4:
                    probs = self.temperature(logits=logits, temperature=10) #10
                    word = self.nucleus(probs=probs, p=0.9) #0.99
                else:
                    flag=1

            # CONDITION TO TACKLE THE "artist:"
            # which seems to pop up if the prompt is empty (?)
            if 'artist' in self.word2event[word]:
                probs = self.temperature(logits=logits, temperature=10) #10
                word = self.nucleus(probs=probs, p=0.99) #0.99

            # CONDITION TO TACKLE THE "end"
            if self.word2event[word] == 'end':
                probs = self.temperature(logits=logits, temperature=10) #5
                word = self.nucleus(probs=probs, p=0.99) #0.99

            #enforce time signature
            new_measure = False
            if 'wait' in self.word2event[word]:
                split_word = self.word2event[word].split(":")
                wait_amnt = int(split_word[1])
                if ticks_since_measure + wait_amnt > ticks_per_measure:
                    print(ticks_per_measure, ticks_since_measure)
                    new_wait_amnt = ticks_per_measure - ticks_since_measure
                    word = self.event2word["wait:" + str(new_wait_amnt)]
                    new_measure = True
                    ticks_since_measure = 0
                elif ticks_since_measure + wait_amnt == ticks_per_measure:
                    new_measure = True
                    ticks_since_measure = 0
                else:
                    ticks_since_measure += wait_amnt
            elif "new_measure" == self.word2event[word]:
                skip_token = True
            elif "tempo" in self.word2event[word]:
                skip_token = True
            
            if not skip_token:
                words[0].append(word)
                final += self.word2event[word] + '\n'
            if new_measure:
                event = 'new_measure'
                words[0].append(self.event2word[event])
                final += event + '\n'
                generate_n_bar += 1
                measures_since_repeat += 1
        
        return final.split('\n')

    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word