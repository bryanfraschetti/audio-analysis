import numpy as np
import os
from . import cqtspec, config_env
import joblib
from . import config_env

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

def projection(num_components, segment_vector, bases):
    projection = np.zeros(num_components)
    for i in range(num_components): #iterate through bases and compute projection
        projection[i] = (np.dot(segment_vector, bases[i])/np.dot(bases[i], bases[i]))
    return projection

def linearCombination(num_components, weights, bases, average_loudness=0):
    lc = np.zeros(bases[0].shape[0]) #dimensionality of basis vectors
    for i in range(num_components):
        lc = lc + weights[i]*bases[i]
    lc = lc + average_loudness #add the average loudness of the segment, if not specified assume 0
    return lc

def recompose_spectrogram(song):
    try:
        model = joblib.load('models/cqt_pca_model_nc' + str(NUM_COMPONENTS) #load current model
                            + "_hs" + str(HOP_LENGTH)
                            + '_fs' + str(F_BINS)
                            + "_ntb" + str(NUM_TIME_BINS)
                            + "_sd" + str(SEGMENT_DURATION)
                            + ".joblib")
        components = model.components_
    except:
        FileNotFoundError("No model for current environment configuration exists")
        return

    Log_power, _ = cqtspec.log_power_spectrogram(song)

    Log_power = cqtspec.truncate_spectrogram(Log_power) #truncate to be integer number of segments

    segmented_spectrogram = cqtspec.segment_spectrogram(Log_power)

    #preprocessing, remember loudness so it can be applied after projection
    scaled_segments, loudnesses = cqtspec.normalize(segmented_spectrogram)

    # calculate projections onto bases

    segmentProjections = []
    for scaled_segment in scaled_segments:
        projectionOnBases = projection(NUM_COMPONENTS, scaled_segment, components[0:NUM_COMPONENTS])
        segmentProjections.append(projectionOnBases)


    # compute linear combination of projections, adding back the loudness
    reconstructed = []
    for index, segmentProjection in enumerate(segmentProjections):
        reconstructed.append(linearCombination(
            NUM_COMPONENTS, 
            segmentProjection, 
            components[0:NUM_COMPONENTS], 
            loudnesses[index])
        )

    #reshape reconstruction to spectrogram dimensions
    reconstructed_spectrogram = cqtspec.array_to_spectrogram_shape(np.array(reconstructed))

    return reconstructed_spectrogram