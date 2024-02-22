from src import cqtpca, cqtrecon, cqtspec, config_env
import librosa
import soundfile as sf
import os
import joblib
import numpy as np

repo_root = os.path.abspath(os.path.join(__file__, ".."))
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

model = joblib.load('models/cqt_pca_model_nc' + str(NUM_COMPONENTS) #load current model
                    + "_hs" + str(HOP_LENGTH)
                    + '_fs' + str(F_BINS)
                    + "_ntb" + str(NUM_TIME_BINS)
                    + "_sd" + str(int(1000*SEGMENT_DURATION))
                    + ".joblib")

print(np.sum(model.explained_variance_ratio_))