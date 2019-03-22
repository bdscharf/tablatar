import sys
from Song import Song
import librosa
import numpy as np
import pandas as pd
import os.path

'''
    Main driver for extracting notes and tablature from a file path.
'''
def main():
    if len(sys.argv) == 2: 
        input_path = sys.argv[1]
        print('Input file staff: ')
        MainSong = Song(file_path=input_path)
        MainSong.extract_notes(file_path=input_path, chunk=False)
        MainSong.map_strings('open_previous')
        staff = MainSong.generate_lilypond()
        print(staff)

    # print('Generating tablature for all files in onset_comparison/sample_recordings...')
    # recordings_df = pd.read_csv('./onset_comparison/sample_recordings.csv')

    # for index, row in recordings_df.iterrows():
    #     current_path = './onset_comparison/samples/' + row['Filename'] + '.wav'
    #     real_onsets = row['actual # notes']
    #     currentSong = Song()
    #     currentSong.extract_notes(file_path=current_path, chunk=False)
    #     currentSong.map_strings(heuristic='open_previous')
    #     currentSong.generate_lilypond(file_name='./output_staff/'+row['Filename'])

if __name__ == '__main__':
    main()