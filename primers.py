prompt_e = ['artist:unknown_artist', 'downtune:0', 'tempo:' + "BPM_PLACEHOLDER", 'start', 'new_measure',
                'distorted0:note:s6:f0',
                'bass:note:s5:f0',
                'drums:note:36',
                'drums:note:42',
                'wait:240']

prompt_a = ['artist:unknown_artist', 'downtune:0', 'tempo:' + "BPM_PLACEHOLDER", 'start', 'new_measure',
                'distorted0:note:s5:f0',
                'bass:note:s4:f0',
                'drums:note:36',
                'drums:note:42',
                'wait:240']

prompt_d = ['artist:unknown_artist', 'downtune:0', 'tempo:' + "BPM_PLACEHOLDER", 'start', 'new_measure',
                'distorted0:note:s4:f0',
                'bass:note:s3:f0',
                'drums:note:36',
                'drums:note:42',
                'wait:240']

prompt_empty = ['artist:unknown_artist', 'downtune:0', 'tempo:' + "BPM_PLACEHOLDER", 'start', 'new_measure']

primer_dict = {
        1: (prompt_e, 240),
        2: (prompt_a, 240),
        3: (prompt_d, 240),
        4: (prompt_empty, 0)
}

def get_primer_prompt(primer_id, bpm):
        primer, offset = primer_dict[primer_id]
        primer[2] = 'tempo:' + str(bpm)
        return primer, offset