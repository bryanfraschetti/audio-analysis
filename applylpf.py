from src import lpf
import librosa
import soundfile as sf

song, sr = librosa.load("Reconstructions/VIVA_nc200_fs108_FPS17_reconstructed.wav", sr=44100)

cutoff = 11000

song_lpf = lpf.butter_lowpass_filter(song, cutoff=cutoff, fs=44100, order=4)

sf.write("Reconstructions/LPF_VIVA_.wav",song_lpf,44100)

