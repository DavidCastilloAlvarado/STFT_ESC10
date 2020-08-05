#%%
import librosa
import IPython.display as ipd
data, samplerate = librosa.load("dataset/001 - Dog bark/1-30344-A.ogg", sr=44000) 
ipd.Audio(data, rate=samplerate)

# %%
