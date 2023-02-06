from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
import librosa
model = keras.models.load_model('/Volumes/Adithya KP Main/Adithya KP/5th Sem BE/PW Machine Learning/model.h5')


path = "/Volumes/Adithya KP Main/Adithya KP/5th Sem BE/PW Machine Learning/Data/genres_original/blues/blues.00001.wav"
y , frequency = librosa.load(path)
sample_rate = frequency

length = 0
y_trimmed, _ = librosa.effects.trim(y = y, top_db = 35)
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref = np.max)

spectral_centroid = librosa.feature.spectral_centroid(y = y)
chroma_stft = librosa.feature.chroma_stft(y = y)

FRAME_LENGTH = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
rms = librosa.feature.rms(y = y)

mfcc = librosa.feature.mfcc(y = y, n_mfcc = 20)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y = y, sr = frequency)


print(type(spectral_centroid))
print(type(chroma_stft))
print(type(rms))
print(type(mfcc))
print(type(spectral_bandwidth))
Spectral_centroid_mean = np.mean(spectral_centroid, dtype = np.float64)
Spectral_centroid_var = np.var(spectral_centroid, dtype = np.float64)
chroma_stft_mean = np.mean(chroma_stft, dtype = np.float64)
chroma_stft_var = np.var(chroma_stft, dtype = np.float64)
rms_mean = np.mean(rms)
rms_var = np.var(rms)
mfcc_mean = np.mean(mfcc, axis = 1)
mfcc_var = np.var(mfcc, axis = 1)
spectral_bandwidth_mean = np.mean(spectral_bandwidth, axis = 1)
spectral_bandwidth_var = np.var(spectral_bandwidth, axis = 1)
print(Spectral_centroid_mean)
print(Spectral_centroid_var)
print(chroma_stft_mean)
print(chroma_stft_var)
print(rms_mean)
print(rms_var)
print(mfcc_mean)
print(mfcc_var)
print(spectral_bandwidth_mean)
print(spectral_bandwidth_var)
