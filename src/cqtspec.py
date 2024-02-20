import librosa
import numpy as np
from sklearn.preprocessing import scale
import os
from . import config_env

repo_root = os.path.abspath(os.path.join(__file__, "../.."))
env = os.path.abspath(os.path.join(repo_root, "environmentVars.ini"))
config_env.load_ini_env(env)

HOP_LENGTH = int(os.environ.get("hop_length"))
SAMPLE_RATE = int(os.environ.get("sample_rate"))
SEGMENT_DURATION = float(os.environ.get("segment_duration"))
F_BINS = int(os.environ.get("F_BINS"))
BINS_PER_OCTAVE = int(os.environ.get("bins_per_octave"))
NUM_COMPONENTS = int(os.environ.get("num_components"))
F_MIN = float(os.environ.get("f_min"))

NUM_TIME_BINS = int( SEGMENT_DURATION / ( HOP_LENGTH / SAMPLE_RATE) )
FLOATS_PER_SEGMENT = NUM_TIME_BINS * F_BINS

def log_power_spectrogram(song):
    C_song = librosa.cqt(song, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_bins=F_BINS, bins_per_octave=BINS_PER_OCTAVE, fmin=F_MIN) #cqt
    phase = np.angle(C_song)
    Song_power = np.abs(C_song) ** 2 #power is proportional to amplitude squared
    Log_power = librosa.power_to_db(Song_power) #human perception of loudness is logarithmic
    return (Log_power, phase)

def log_power_to_stft(Log_power):
    Song_power = librosa.db_to_power(Log_power) #db back to power
    C_song = np.sqrt(Song_power) #power to amplitude
    return C_song #this is the original stft

def truncate_spectrogram(spectrogram):
    spectrogram = spectrogram.T
    spectrogram = spectrogram.flatten()
    toRemove = len(spectrogram) % FLOATS_PER_SEGMENT
    if toRemove != 0:
        spectrogram = np.delete(spectrogram, np.s_[-toRemove:])
    return spectrogram

def segment_spectrogram(truncated_spectrogram):
    segmented_spectrogram = np.array_split(truncated_spectrogram, len(truncated_spectrogram) / FLOATS_PER_SEGMENT)
    return segmented_spectrogram

def normalize(segmented_spectrogram):
    loudnesses = []
    scaled_segments = []
    for segment in segmented_spectrogram:
        segment = segment.astype('float')
        loudness = np.mean(segment)
        loudnesses.append(loudness)
        scaled_segment = scale(segment, with_std=False) #keep the spread/dispersion of loudness
        scaled_segments.append(scaled_segment)
    return (scaled_segments, loudnesses)

def array_to_spectrogram_shape(arr):
    arr = arr.flatten()
    num_samples = len(arr)
    reshaped_arr = arr.reshape(int(num_samples/F_BINS), F_BINS).T
    return reshaped_arr

def LogPowerPhase_to_ComplexSpectrogram(log_power_spectrogram, phase_spectrogram):
    #combine magnitude and phase information
    mag_spectrogram = log_power_to_stft(log_power_spectrogram)
    recover_complex_rectangular_form = np.vectorize(lambda mag, phi: complex(mag * np.cos(phi), mag * np.sin(phi)))
    reconstructed_spectrogram = recover_complex_rectangular_form(mag_spectrogram, phase_spectrogram)
    return reconstructed_spectrogram
