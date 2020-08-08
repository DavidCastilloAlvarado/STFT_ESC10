# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import librosa.display
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa    
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
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
# new branchDSDSDSDJNJNJNJ  vggvgvFDFDFDdsdsdsdsd
# segundo encabezado
#get_ipython().run_line_magic('matplotlib', 'inline')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
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
import time
time_init = time.time()
rcParams['figure.figsize'] = 20, 5
example = 305
win_length=1024
hop_length=int(win_length/2)
freq = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data[example],n_fft=1024, win_length=win_length ,hop_length=hop_length)), ref=np.max) #44000
##mfcc = librosa.feature.mfcc(y=audio_data[example])
print(freq.shape)
print("Clase: ",labels[example])
print("Windowing Time: ", win_length*1/44000,"s")
time_final = time.time()
data = (time_final- time_init )/1000
print("tiempo de ejecución: ",data)

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

#%%
print(set(labels))


# %%
#Modelo con SVM
from sklearn import tree, svm
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
#from skmultilearn.model_selection import iterative_train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_stft,Y_label, test_size=0.20,random_state =2,stratify= Y_label)
print("Split ready")
clf = svm.SVC(verbose= True,random_state=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM: acc:"+ str(accuracy_score(y_test, y_pred)))

#%%
# Modelo con PCA y SVM
from sklearn.decomposition import PCA
print("Cantidad de caracteristicas", len(X_stft[0]))
pca = PCA(n_components=300)
pca.fit(normalized_stft) ## justa para todo el espectro de datos
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)
clf_PCA = svm.SVC(verbose= True,random_state=2, )
clf_PCA.fit(X_train_PCA, y_train)
y_pred = clf_PCA.predict(X_test_PCA)
print("SVM+PCA: acc:"+ str(accuracy_score(y_test, y_pred)))

#%%
# Modelo XGBoost & PCA
# xgb_model = XGBClassifier(learning_rate=0.01,
#                     n_estimators=1200,
#                     max_depth=600,
#                     min_child_weight=.05,
#                     gamma=0,
#                     subsample=.5,
#                     colsample_bytree=0.5,
#                     objective='multi:softmax',
#                     nthread=4,
#                     num_class=10,
#                     num_parallel_tree = 18,
#                     seed=27,verbosity= 1,n_jobs=8 )
#xgb_model.load_model('models/xgbmodel')
xgb_model = XGBClassifier(learning_rate=0.01,
                    n_estimators=1200,
                    max_depth=100,
                    min_child_weight=.05,
                    gamma=0,
                    subsample=.5,
                    colsample_bytree=0.5,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=10,
                    num_parallel_tree = 18,
                    seed=27,verbosity= 1,n_jobs=8 )
xgb_model.fit(X_train_PCA, y_train)
#xgb_model.save_model('models/xgbmodel')
y_pred = xgb_model.predict(X_test_PCA)
print("XGB+PCA: acc:"+ str(accuracy_score(y_test, y_pred)))

#%%
y_pred = xgb_model.predict(X_test_PCA)
print("XGB+PCA: acc:"+ str(accuracy_score(y_test, y_pred)))
# y_pred = xgb_model.predict(X_test_PCA)
# y_train_aft = xgb_model.predict(X_train_PCA)
# print("XGB+PCA: acc:"+ str(accuracy_score(y_test, y_pred)))
# print("XGB+PCA: train:"+ str(accuracy_score(y_train, y_train_aft)))

# %%
X_train, X_test, y_train, y_test = train_test_split(normalized_stft,Y_label, test_size=0.20,random_state =2,stratify= Y_label)

X_train_NN= np.reshape(X_train,(X_train.shape[0],513, -1,1))
X_test_NN = np.reshape(X_test,(X_test.shape[0],513, -1,1))
y_train_NN=keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test_NN =keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')


# %%
# 1 perifoneo 1 parlante y 6--8jacks / calble UTP6A 1 rollo
# tiempo de entrega
# Compra para compra de proyecto de IVA
model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=X_train_NN.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#'''
#model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

#model.add(Dropout(0.2))

#'''
#'''
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#'''


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

#model.add(Dense(1000))#input_shape=features.shape[1:]
model.add(Dense(64))#input_shape=features.shape[1:]
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# %%

history = model.fit(X_train_NN, y_train_NN,
                    batch_size=8, 
                    epochs=10,
                    validation_data = [X_test_NN,y_test_NN] )

# %%
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation accuracy')
plt.legend(loc=0) 
plt.figure()
plt.show()


# %%
