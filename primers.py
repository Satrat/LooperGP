HEADER1 = ['artist:unknown_artist', 'downtune:0']
HEADER2 = ['start', 'new_measure']

DISTORTED = "distorted0"
BASS = "bass"
DRUMS = "drums"

def build_pitched_note(instrument, string, fret):
        return "{}:note:s{}:f{}".format(instrument, string, fret)

def build_percussion_note(instrument, midi_num):
        return "{}:note:{}".format(instrument, midi_num)

def build_primer(bpm, key=None, duration=0):
        primer = []
        for val in HEADER1:
                primer.append(val)
        primer.append("tempo:{}".format(bpm))
        for val in HEADER2:
                primer.append(val)
        
        if not key == None:
                if key == 'e':
                        primer.append(build_pitched_note(DISTORTED, 6, 0))
                        primer.append(build_pitched_note(BASS, 5, 0))
                elif key == 'a':
                        primer.append(build_pitched_note(DISTORTED, 5, 0))
                        primer.append(build_pitched_note(BASS, 4, 0))
                elif key == 'd':
                        primer.append(build_pitched_note(DISTORTED, 4, 0))
                        primer.append(build_pitched_note(BASS, 3, 0))
                else:
                        print("Unrecognized key {}, defaulting to e".format(key))
                        primer.append(build_pitched_note(DISTORTED, 6, 0))
                        primer.append(build_pitched_note(BASS, 5, 0))
                primer.append(build_percussion_note(DRUMS, 42))
                primer.append(build_percussion_note(DRUMS, 36))

                primer.append("wait:{}".format(duration))
        
        return primer
                

#def get_primer_prompt(primer_id, bpm):
#        primer, offset = primer_dict[primer_id]
#        primer[2] = 'tempo:' + str(bpm)
#        return primer, offset