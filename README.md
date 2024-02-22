# A Repository for Audio Analysis and PCA Feature Extraction

This branch uses the STFT (Short Time Fourier Transform) which is based on linear frequency binning, which means the audio spectrum from 20Hz to 20kHz is divided into frequency bins of equal bandwidth.
The CQT (Constant Q Transform) branch uses non-linear frequency binning, where the frequency bands are proportional to the size of the frequency. This allows for better frequency resolution in the low range (20Hz to 2kHz), which contains most of the waveform energy. Applying PCA on the CQT yields better qualitative results for the same number of basis functions.

## Pipeline:

### I. Obtain spectrogram

First load a .wav file or .mp3 and convert it to a temp .wav using pydubs and mktemp. WAV files use PCM samples at a rate of >44kHz to encode audio. Compile N consecutive samples to be one frame (approximate time instant) and apply a Hann window on the frame as a preprocessing method to remove high frequency artifacts and compute the Fast Fourier Transform (FFT) on this frame. This process is known as the STFT, as it localizes the Discrete Fourier Transform (DFT) to the frame. After applying the local FFT, hop to next frame, which overlaps with current frame such that no information is lost due to the windowing process.

### II. Compute power spectrogram and log power spectrogram

The result of the previous step is called a spectrogram and it represents the amplitudes of all frequency bins at all time bins. Since it represents the amplitude of a sine wave it can be converted to Power ∝ A², and moreover to volume in dB Pᵈᴮ=10 log(P). The decibel representation is the most meaningful since it most strongly correlates with the human perception of audio which is that a 10 fold increase in magnitude corresponds to the perception of a linear increase.

### III. Partition spectrogram into segments of fixed length

The spectrogram is to be partitioned into time intervals of fixed duration. Depending on the application this is configurable. Psychoacoustics indicate that audio within the same 50ms interval fuse to be the same perceptual event, meaning humans hear all sounds within a 50ms interval simultaneously. So to not lose significant reconstruction, the segments should not exceed 50ms in duration. N samples @ fs = 44.1kHz corresponds to a time of T = N/44.1kHz. So while increasing N provides more frequency resolution, it decreases time resolution within the 50ms segment. For a frame size N, there will be [(N/2) + 1] frequency bins and floor(0.05*44100/[(N/2)+1]) time bins. The number of data entries per segment is then D=NumberOfFrequencyBins * NumberOfTimeBins.

### IV. Reshape partitioned spectrograms into vectors for PCA processing

The spectrotemporal surface (spectrogram) is 3 dimensional, a 2D matrix where one axis corresponds to the frequency, the other time, and the amplitude is the volume of that frequency at that time within the segment. The quality and characteristic of sound is independent of the segments average volume but is rather related to the shape of the surface relative to the average volume. It should therefore be normalized zero mean, which can be interpreted mathematically as "subtracting the average volume from each frequency-time bin" or physically as thinking of the average volume as the new origin. Moreover, the segment can be scaled by the standard deviation. If the spectrotemporal surface is considered a vector, scaling by the standard deviation does not change the direction of the vector, only the magnitude such that it is comparable to a unit vector (zero-mean, std-dev=1). For PCA it is also convenient to flatten the 2D matrix into a 1D vector with DataEntries number of entries (dimensions), which can later be reshaped back into the original segment spectrogram shape.

### V. Perform PCA on the reshaped vectors to compute the n most important basis functions

If the spectrogram shape has D entries, it will flatten to a D dimensional (1xD vector). All 50ms segments can be thought of as existing in this D dimensional space. We want to represent this space with less dimensions by using NumComponents basis dimensions oriented within the D dimensional space such that they maximize the variance over the dataset. This is equivalent to saying that the new basis functions best describe the original pointset, minimizing the amount of data lost. Interestingly these basis sounds will be representative of the most common "directions" of audio.

### VI. Convert the vectors back into spectrotemporal surfaces and plot them

The resulting basis functions/components are 1xD vectors in the original space and can be reshaped to the frequency-time bin matrix.

### VII. Convert audio to the lower dimensional space

Any song can be converted to its STFT, which can then be projected onto the basis functions. This indicates how much the song aligns with each basis function. The point in the D-dimensional space can then be computed as the linear combination of the projections on the D-dimensional bases. There will be some loss since you are representing a D-dimensional point with NumComponents directions and D>NumComponents. Then all that needs to be done is inverse the transform

## Usage

    Place training data in resources
    Configure environment specs in environmentVars.ini
        - frame_size: affects how many samples are used for each STFT. Larger frame_size is better frequency resolution
            but worse time resolution
        - frames_per_segment: how many frames (stfts) compose your basis spectrotemporal surfaces. more
            frames_per_segment means each basis function will represent a longer unit of time
        - sample_rate: sample rate used for audio processing
        - num_pca_components: how many basis functions are to be used

## Examples

`main.py` contains some examples of things you might want to use this for such as

    1. Performing PCA
    2. Visualizing pca basis functions
    3. Going from song directly to log power spectrogram
    4. Projection of a song onto bases and reconstruction of log power spectrogram
    5. Comparing the log power spectrogram of the original song with the reconstructed log power spectrogram (RMSE,MAE)
    6. Converting magnitude spectrogram back to audio (using phase spectrogram for timing recovery)

## Summary of modules

### src/pca_decomposition.py:

Handles PCA decomposition. All thats required is a function call to pca_decomposition.generate_pca_bases(). Optional parameters are:

    plot_bases, which plots the basis functions on computation
    save_bases_as_audio, which saves the bases as .wav files in bases/pca_{model_identification}/component[i].wav

### src/pca_recomposition.py:

Performs vector projection onto basis functions, computes the linear combination, and reconstructs the
magnitude spectrogram

### src/spectrogram_operations.py:

Has functions that simplify operations relating to spectrograms:

    plot_spectrogram(spectrogram, etc) plots the passed spectrogram and configures the plot using other params \

    log_power_spectrogram(song) converts a song directly to the logarithmic power spectrogram \

    log_power_to_stft(spectrogram) takes a logarithmic power spectrogram and turns it back into the magnitude
    spectrogram \

    roundtrip(song) takes a song, computes the log power spectrogram, converts back to the complex spectrogram,
    then inverses the stft to convert it back into a song. This isn't particularly useful except to see
    any effect/loss of data due to: stft quantizations, framing, windowing, and perhaps even
    numeric (float) error since the data cannot be represented with 100% precision \

    truncate_spectrogram(spectrogram) shortens the spectrogram by less than one segment to ensure the
    spectrogram fits evenly into the fundamental time unit determined by the segment duration \

    segment_spectrogram(spectrogram) splits the spectrogram into segments (time units) \

    normalize(segment) takes a segment and normalizes it by the average loudness since the bases
    were intentionally constructed to be independent of loudness. The average loudness is stored so that
    it can later be recalled and recombined with the projections to preserve the original loudness \

    array_to_spectrogram_shape(arr) takes a flattened array and reshapes it to have the same number of frequency
    bins determined by the pca_model. The remaining dimension is equal to the number of stfts (frames) that
    fit into the array. This is useful because the spectrograms are often flattened to 1D vectors for purposes
    such as projection and truncation, but need to be converted back to the spectrogram shape if they are to
    be visualized \

    logpowerphase_to_complexspectrogram(log_power,phase) combines log power and phase information to recover the
    original magnitude spectrogram (stft of the original signal) \

    plot_pca_bases(): (default) plots the components of the current environment configuration, but can be overloaded
    to plot any models bases with matplotlib formatting (num_rows, num_cols, components)
