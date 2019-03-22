import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

recording_df = pd.read_csv('./sample_recordings.csv')

p_diffs = []

# check for normalization

for index, row in recording_df.iterrows():
    file_name = row['Filename'] + '.wav'
    actual_count = row['actual # notes']

    y, sr = librosa.load('./samples/' + file_name)

    hop = 512
    wait = ((0.5 * sr) / hop)
    # librosa.util.peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait) 
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean)
    onsets = librosa.onset.onset_detect(y, sr, hop_length=hop, onset_envelope=onset_env, wait=wait, units='samples', backtrack = True)
    #onsets = librosa.onset.onset_detect(y, sr, hop_length=hop, wait=wait, units='samples')

    # o_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
    # times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    # onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, wait=wait)


    predicted_count = len(onsets)
    recording_df.ix[index, 'detected # notes'] = predicted_count

    diff = abs(actual_count - predicted_count)
    percent_diff = diff / actual_count
    p_diffs.append(percent_diff)

    print('Actual count: ' + str(actual_count))
    print('Predicted count: ' + str(predicted_count))
    print('Difference of ' + str(diff) + ' in ' + file_name)
    print('Percentage difference of ' + str(percent_diff))
    print('---')

recording_df.to_csv('./sample_recordings_detected.csv')

print('Average difference: ' + str(np.average(p_diffs)))

# y, sr = librosa.load('./bs_2019_samples/acoustic_5.wav')

# o_env = librosa.onset.onset_strength(y, sr=sr)
# times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
# onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

# print(len(onset_frames))

# D = np.abs(librosa.stft(y))
# plt.figure()
# ax1 = plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log')
# plt.title('Power spectrogram')
# plt.subplot(2, 1, 2, sharex=ax1)
# plt.plot(times, o_env, label='Onset strength')
# plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
# plt.axis('tight')
# plt.legend(frameon=True, framealpha=0.75)

# plt.show()