import librosa
import numpy as np
import os
from . import cqtspec, config_env
from sklearn.decomposition import PCA
import joblib

repo_root = os.path.abspath(os.path.join(__file__, "../.."))
env = os.path.abspath(os.path.join(repo_root, "environmentVars.ini"))
config_env.load_ini_env(env)

HOP_LENGTH = int(os.environ.get("hop_length"))
SAMPLE_RATE = int(os.environ.get("sample_rate"))
SEGMENT_DURATION = float(os.environ.get("segment_duration"))
F_BINS = int(os.environ.get("f_bins"))
BINS_PER_OCTAVE = int(os.environ.get("bins_per_octave"))
NUM_COMPONENTS = int(os.environ.get("num_components"))
F_MIN = float(os.environ.get("f_min"))

NUM_TIME_BINS = int( SEGMENT_DURATION / ( HOP_LENGTH / SAMPLE_RATE) )
FLOATS_PER_SEGMENT = NUM_TIME_BINS * F_BINS

def generate_pca_bases():
    #read songs
    songlist = []
    for root, _, files in os.walk("Resources/"):
        for file in files:
            if file.endswith(".wav"):
                song, _ = librosa.load(root + file)
                songlist.append(song)

    # compute log power spectrogram for all data, then divide into segments 
    segments = [] #collection of all 50ms intervals
    for song in songlist:
        power_spectrogram, _ = cqtspec.log_power_spectrogram(song)
        truncated_spectrogram = cqtspec.truncate_spectrogram(power_spectrogram) #fit data to schema
        segmented_spectrogram = cqtspec.segment_spectrogram(truncated_spectrogram) #cut into segments
        for segment in segmented_spectrogram:
            segments.append(segment) #append each segment to the master segment list

    segments = np.array(segments)
    segments = segments.astype("float")
    normalized_data = cqtspec.normalize(segments)[0] #normalize loudness

    model = PCA(n_components=NUM_COMPONENTS) #perform pca on dataset
    model.fit_transform(normalized_data)

    joblib.dump(model, 'models/cqt_pca_model_nc' + str(NUM_COMPONENTS) #save model and include parameters in title
                + "_hs" + str(HOP_LENGTH)
                + '_fs' + str(F_BINS) 
                + "_ntb" + str(NUM_TIME_BINS)
                + "_sd" + str(SEGMENT_DURATION)
                + ".joblib")