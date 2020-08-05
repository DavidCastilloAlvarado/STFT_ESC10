# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa    
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import tensorflow.keras as keras
import pandas as pd
from pylab import rcParams
import random
rcParams['figure.figsize'] = 20, 5
# encabezado
# tercer encabezado
# new branch
# segundo encabezado
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
audio_data = []
labels = []
sampling_rate = []
file_names = []


# %%
data, samplerate = librosa.load("dataset/001 - Dog bark/1-30344-A.ogg", sr=44000) 
print(data.shape)
print(samplerate)


# %%
time_sec = (len(data)/samplerate)
step = time_sec/len(data)
print("Duración: ", time_sec,"s")
print("SampleTime: {:6f}s ".format(step))
i=0
time_divion=[]
while i<=time_sec-step:
    
    time_divion.append(i)
    i=i+step
# the fourth second step
four_sec_step_number = (4*len(time_divion))/time_sec
print(four_sec_step_number)


# %%
classes = []
label_number=0
audio_data = []
labels = []
sampling_rate = []
file_names = []
data = []
noisy_removed=[]
noise=[]
for filepath in glob.iglob('dataset/*'):
    #print(filepath[9:])
    #print(filepath)
    classes.append(filepath[8:])
print(classes)

for i in classes:
    print("the class = "+i+", the label = "+str(label_number))
    for j in glob.iglob('dataset/'+i+'/*'):
        #samplerate, data = wavfile.read(j)
        y, s = librosa.load(j, sr=44000) # Downsample 44.1kHz
        #reduced_noise = nr.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)
        #print(s)
        #print(j)
        data.append([y,label_number])
        #noise.append(y)
        #labels.append(label_number)
        
    label_number = label_number + 1
#print(len(labels))
print(label_number)
# Data is now the list with the whole resample audio to 44Khz


# %%
# Chocolatea toda la data existente dentro de la lista "data"
#       Coloca los nuevos datos chocolateados dentro de audio_data y labels
import random
random.shuffle(data)
for i,j in data:
    audio_data.append(i)
    labels.append(j)


# %%
print(len( set(labels) ))


# %%
example = random.randint(0,400)


# %%
save_path='Helicopter_before.jpg'
print("Time divisions: ",len(audio_data[example]))
plt.plot(time_divion[0:192000],audio_data[example][0:192000])
#plt.show()
plt.title('signal in real time')
pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
#pylab.close()


# %%
from pylab import rcParams
import random
rcParams['figure.figsize'] = 20, 5
example = 305
win_length=1024
hop_length=int(win_length/2)
freq = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data[example],n_fft=1024, win_length=win_length ,hop_length=hop_length)), ref=np.max) #44000
mfcc = librosa.feature.mfcc(y=audio_data[example])
print(freq.shape)
print("Clase: ",labels[example])
print("Windowing Time: ", win_length*1/44000,"s")

save_path = 'Helicopter.jpg'
#plt.axis('off') # no axis
librosa.display.specshow(freq,hop_length=hop_length,x_axis='time', y_axis='linear',sr=44000)
plt.colorbar(format='%+2.0f dB')
plt.title('short time fourier transform')
#pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
#pylab.close()


# %%
import IPython.display as ipd
from scipy.io import wavfile
wavfile.write('test.wav', 44000, audio_data[example])

# %%
X_stft = []
Y_label=[]
from tqdm import tqdm
for i in tqdm(range (0,400)):
    # if i%100 == 0:
    #     print(i)
    freq = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data[i],n_fft=1024, win_length=win_length ,hop_length=hop_length)), ref=np.max)

    #freq = np.abs(librosa.stft(audio_data[i], n_fft=512, hop_length=256, win_length=512))

    freq=freq.reshape(-1,1)
    #print(freq.shape)
    if freq.shape[0]==220590:
        X_stft.append(freq)
        Y_label.append(labels[i])
    
X_stft =np.stack(X_stft) 
X_stft.shape


# %%
X_stft=X_stft.reshape(381,220590)
X_stft.shape


# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_stft,y = X_stft)

normalized_stft = scaler.transform(X_stft)

print(np.amax(X_stft))
print(np.amax(normalized_stft))


# %%
from sklearn import tree
X_train, X_test, y_train, y_test = train_test_split(X_stft, Y_label, test_size=0.20, random_state=1150)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# %%
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


# %%
features_convolution = np.reshape(normalized_stft,(381,513, -1,1))
features_convolution.shape

# %%
y=keras.utils.to_categorical(Y_label, num_classes=10, dtype='float32')


# %%
model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=features_convolution.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#'''
#model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.2))

#'''
#'''
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#'''


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

#model.add(Dense(1000))#input_shape=features.shape[1:]
model.add(Dense(64))#input_shape=features.shape[1:]

model.add(Dense(10))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# %%
history = model.fit(features_convolution, y,batch_size=8, epochs=40,validation_split=0.2)


# %%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
