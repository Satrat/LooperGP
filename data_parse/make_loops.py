import guitarpro
import sys
import dadagp as dada
import numpy as np
import os

def convert_from_dadagp(tokens):
    song = dada.tokens2guitarpro(tokens, verbose=False)
    song.artist = tokens[0]
    song.album = 'Generated by DadaGP'
    song.title = "untitled"
    return song

class MelodyNote:
    def __init__(self, duration, start, bar_start, note_list):
        self.duration = duration.value
        self.is_dotted = duration.isDotted
        self.tick_duration = 3840.0 / self.duration #3840 ticks in whole note
        if self.is_dotted:
            self.tick_duration = self.tick_duration * 1.5
        
        self.start_time = start
        self.on_bar = False
        if self.start_time == bar_start:
            self.on_bar = True

        self.notes = ["0:0"]
        self.note_types = [guitarpro.NoteType.rest]
        if len(note_list) > 0: #not a rest
            self.notes = set([f"{n.string}:{n.value}" for n in note_list])
            self.note_types = set([n.type for n in note_list])

    def __str__(self):
        return f"{self.duration} {self.is_dotted} {self.notes} {self.note_types} at {self.start_time}"
    
    def __eq__(self, other):
        if self.duration != other.duration:
            return False
        if self.is_dotted != other.is_dotted:
            return False
        
        if len(self.notes) != len(other.notes):
            return False
        for m in self.notes:
            if m not in other.notes:
                return False
        
        return True
    
def compare_patterns(p1, p2): #new pattern, existing pattern
    if len(p1) < len(p2):
        for i in range(len(p1)):
            if p1[i] != p2[i]:
                return 0 #not a substring, theres a mismatch
            return 1 #is a substring
    else:
        for i in range(len(p2)):
            if p1[i] != p2[i]:
                return 0 #not a substring, theres a mismatch
            return 2 #existing pattern is substring of the new one, replace it

def test_loop_exists(pattern_list, pattern):
    for i, pat in enumerate(pattern_list):
        result = compare_patterns(pattern, pat)
        if result == 1:
            return -1 #ignore this pattern since its a substring
        if result == 2:
            return i #replace existing pattern with this new longer one
    return None #we're just appending the new pattern

def create_track_list(song):
    melody_track_lists = []
    time_signatures = {}
    for i, track in enumerate(song.tracks):
        melody_list = []
        for measure in track.measures:
            for beat in measure.voices[0].beats:
                note = MelodyNote(beat.duration, beat.start, measure.start, beat.notes)
                melody_list.append(note)
            if i == 0:
                signature = (measure.timeSignature.numerator, measure.timeSignature.denominator.value)
                if signature in time_signatures.keys():
                    time_signatures[signature] += 1
                else:
                    time_signatures[signature] = 1
        melody_track_lists.append(melody_list)
        
    return melody_track_lists, time_signatures

def get_dom_beats_per_bar(time_signatures):
    max_repeats = 0
    dom_sig = None
    for k,v in time_signatures.items():
        if v > max_repeats:
            max_repeats = v
            dom_sig = k
    
    num, dem = dom_sig
    ratio = 4.0 / dem
    return num * ratio

def calc_correlation(track_list, instrument):
    melody_seq = track_list[instrument]
    corr_size = len(melody_seq)
    corr_mat = np.zeros((corr_size, corr_size), dtype='int32')
    corr_dur = np.zeros((corr_size, corr_size), dtype='float')

    for j in range(1, corr_size):
        if melody_seq[0] == melody_seq[j]:
            corr_mat[0,j] = 1
            corr_dur[0,j] = melody_seq[j].tick_duration
        else:
            corr_mat[0,j] = 0
            corr_dur[0,j] = 0
    for i in range(1, corr_size-1):
        for j in range(i+1, corr_size):
            if melody_seq[i] == melody_seq[j]:
                corr_mat[i,j] = corr_mat[i-1, j-1] + 1
                corr_dur[i, j] = corr_dur[i-1, j-1] + melody_seq[j].tick_duration
            else:
                corr_mat[i,j] = 0
                corr_dur[i,j] = 0
    
    return corr_mat, corr_dur, melody_seq

def get_valid_loops(melody_seq, corr_mat, corr_dur, min_len=4, min_beats=16.0, max_beats=32.0, min_rep_beats=4.0):
    x_num_elem, y_num_elem = np.where(corr_mat == min_len)
    #print(len(x_num_elem))

    valid_indices = []
    for i,x in enumerate(x_num_elem):
        y = y_num_elem[i]
        start_x = x - corr_mat[x,y] + 1
        start_y = y - corr_mat[x,y] + 1
        
        loop_start_time = melody_seq[start_x].start_time
        loop_end_time = melody_seq[start_y].start_time
        loop_beats = (loop_end_time - loop_start_time) / 960.0
        if loop_beats <= max_beats and loop_beats >= min_beats:
            valid_indices.append((x_num_elem[i], y_num_elem[i]))
    #print(len(valid_indices))
    
    loops = []
    loop_bp = []
    corr_size = corr_mat.shape[0]
    for start_x,start_y in valid_indices:
        x = start_x
        y = start_y
        while x+1 < corr_size and y+1 < corr_size and corr_mat[x+1,y+1] > corr_mat[x,y]:
            x = x + 1
            y = y + 1
        beginning = x - corr_mat[x,y] + 1
        duration = corr_dur[x,y] / 960.0
        end = y - corr_mat[x,y] + 1
        
        if duration >= min_rep_beats and melody_seq[beginning].on_bar:
            loop = melody_seq[beginning:end]
            exist_result = test_loop_exists(loops, loop)
            if exist_result == None:
                loops.append(loop)
                loop_bp.append((melody_seq[beginning].start_time, melody_seq[end].start_time))
            elif exist_result > 0: #index to replace
                loops[exist_result] = loop
                loop_bp[exist_result] = (melody_seq[beginning].start_time, melody_seq[end].start_time)
    #print(len(loops))
    
    return loops, loop_bp

def convert_gp_loops(song, endpoints):
    used_tracks = []
    start = endpoints[0]
    end = endpoints[1]
    for inst in range(len(song.tracks)):
        measures = []
        non_rests = 0
        for measure in song.tracks[inst].measures:
            if measure.start >= start and measure.start < end:
                measures.append(measure)
                for beat in measure.voices[0].beats:
                    for note in beat.notes:
                        if note.type != guitarpro.NoteType.rest:
                            non_rests = non_rests + 1
            else:
                valid_beats = []
                for beat in measure.voices[0].beats:
                    if beat.start >= start and beat.start < end:
                        valid_beats.append(beat)
                        for note in beat.notes:
                            if note.type != guitarpro.NoteType.rest:
                                non_rests = non_rests + 1
                if len(valid_beats) > 0:
                    measure.voices[0].beats = valid_beats
                    measures.append(measure)
        if len(measures) > 0 and non_rests > 0:
            song.tracks[inst].measures = measures + measures + measures + measures + measures + measures + measures + measures
            used_tracks.append(song.tracks[inst])
        if inst == 0 and non_rests == 0: #if the loop is just rests, ignore it
            return None
        
    song.tracks = []
    if len(used_tracks) == 0:
        return None
    for track in used_tracks:
        song.tracks.append(track)
    return song
