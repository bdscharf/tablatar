import pickle
import os.path
import numpy as np
import librosa
import aubio
import itertools
from Evaluator import Evaluator

'''
Song: represents a song and its frettings
'''
class Song:
    '''
    notes = [E2, E4, ...]
    strings = [5, 2]
    mapping = [[note, string, fret], [note, string, fret]]
    Note: no validation currently performed
    '''
    def __init__(self, file_path='', notes=[]):
        self.mapping = []
        self.notes = notes
        
        # Load in fret mappings
        with open("fret_mappings.pkl", "rb") as f:
            self.fret_mappings = pickle.load(f)
        
        # if file_path:
        #     self.extract_notes(file_path)

    def extract_notes(self, file_path, chunk=True):
        #chunk it into chunks where the separation is in between
        audio, sr = librosa.load(file_path, sr=22050)

        hop = 512
        wait = ((0.5 * sr) / hop)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, aggregate=np.mean)
        onsets = librosa.onset.onset_detect(audio, sr, hop_length=hop, onset_envelope=onset_env, wait=wait, units='samples', backtrack = True)

        pitchtracker = aubio.pitch(method="default", buf_size=1024, hop_size=1024, samplerate=22050)
        pitchtracker.set_tolerance(.9)
        notes = []
        group = []
        pause = sr
        groupings = []
        sub = []

        if chunk:
            for i in range(1,len(onsets)):
                if onsets[i] - onsets[i-1] >= pause:
                    groupings.append(sub)
                    sub = [onsets[i]]
                else:
                    sub.append(onsets[i])
                    # print("appended")
            for g in groupings:
                if g == []:
                    continue
                for onset in g:
                    if onset + 1024 < len(audio):
                        frequency = pitchtracker(audio[onset:onset+1024])
                        if frequency <= 1310 and frequency >= 82:
                            note = librosa.core.hz_to_note(frequency)
                            group.append(note[0])
                notes.append(group)
                group = []
        else:
            for onset in onsets:
                if onset + 1024 < len(audio):
                    frequency = pitchtracker(audio[onset:onset+1024])
                    if frequency <= 1310 and frequency >= 82:
                        note = librosa.core.hz_to_note(frequency)
                        notes.append(note[0])

        self.notes = notes
        return notes

    # (very) naive implementation of mapping strings
    # heuristic: 
    #   previous = closest to previous fret
    #   open_previous = prefer open, and then previous fret if not able
    def map_strings(self, heuristic, k=1):
        # Gets the nearest [note, string, fret] combo to target_fret
        def get_nearest(note, target_fret):
            # Get all available options
            options = self.fret_mappings.get(note)
            # Sort by closest fretting, get closet string/fret tuple to target_fret
            nearest = sorted(options, key=lambda option: abs(option[1] - target_fret))[0]
            # Create new mapping element to be appended
            mapping_element = [note.lower(), nearest[0], nearest[1]]
            return mapping_element

        # Make sure average_k is possible
        assert k > 0, '[ALERT] k must be greater than zero.'

        
        # Setting here to make sure we don't lose the value
        perm_heuristic = heuristic

        # Make sure the original note list isn't damaged
        notation = self.notes

        # 'Chunk' if necessary
        if not any(isinstance(e, list) for e in notation):
            notation = [notation]

        # Get closest by heuristic
        for chunk in notation:
            # Handle base case
            first_choices = self.fret_mappings.get(chunk[0])
            if not first_choices:
                first_choices = self.fret_mappings.get('E2')
            all_mappings = []
            for choice in first_choices:
                choice_mapping = [[chunk[0].lower(), choice[0], choice[1]]]
                for index, note in enumerate(chunk[1:]):
                    heuristic = perm_heuristic
                    fret_choices = self.fret_mappings.get(note)
                    if not fret_choices:
                        fret_choices = self.fret_mappings.get('E2')

                    if heuristic == 'average_k' and index >= k-1:
                        open_choice = [choice for choice in fret_choices if choice[1] == 0]
                        if open_choice: # still prefer open strings
                            open_choice = open_choice[0]
                            choice_mapping.append([note.lower(), open_choice[0], open_choice[1]])
                        else:
                            previous_k = [x[2] for x in choice_mapping[index-k+1:]]
                            target_fret = np.average(previous_k)
                            choice_mapping.append(get_nearest(note, target_fret))
                    else:
                        heuristic = 'open_previous'

                    if heuristic == 'open_previous': # fall-through
                        open_choice = [choice for choice in fret_choices if choice[1] == 0]
                        if open_choice:
                            open_choice = open_choice[0]
                            choice_mapping.append([note.lower(), open_choice[0], open_choice[1]])
                        else:
                            heuristic = 'previous'
                    # Fall through to previous if no open found
                    if heuristic == 'previous':
                        choice_mapping.append(get_nearest(note, choice_mapping[index - 1][2]))

                # Append this particular mapping option        
                all_mappings.append(choice_mapping)

            best_mapping = max(all_mappings, key=lambda x: Evaluator(x).total_distance())
            self.mapping.extend(best_mapping)

    def generate_lilypond(self, file_name=''):
        def convert_octave(octave):
            # All octaves relative to C4
            if octave >= 4:
                return '\'' * (octave - 3)
            elif octave < 4:
                return ',' * (3 - octave)
            else:
                return ''

        # Generate staff
        staff = '%' + str(self.notes) + '\n'
        staff += '\\new TabStaff { '
        for note in self.mapping:
            full_note = note[0]
            note_name = full_note[0]
            if '#' in full_note:
                octave = full_note[2]
            else:
                octave = full_note[1]

            octave = convert_octave(int(octave))
            note_string = str(note[1])

            if '#' in full_note:
                staff += note_name + 'is'
            else:
                staff += note_name
            staff += octave

            staff += '\\' + str(note_string) + ' '

        staff += '}'

        if file_name:
            with open(file_name + '.ly', 'w') as f:
                f.write(staff)
            print('Staff written to ' + file_name + '.ly')

        return staff