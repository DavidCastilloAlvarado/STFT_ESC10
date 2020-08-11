# %%
from IPython import get_ipython
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow.keras as keras
import pandas as pd
from pylab import rcParams
import random
import time
from tqdm import tqdm

rcParams['figure.figsize'] = 20, 5
#get_ipython().run_line_magic('matplotlib', 'inline')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
else:
    print("There are not GPUs avaliable")
# %%
audio_data = []
labels = []
sampling_rate = []
file_names = []
classes = []
label_number=0
audio_data = []
labels = []
sampling_rate = []
file_names = []
data = []
noisy_removed=[]
noise=[]
samplerate = 44000
for filepath in glob.iglob('dataset/*'):
    #print(filepath[9:])
    #print(filepath)
    classes.append(filepath[8:])
print(classes)

for i in classes:
    print("the class = "+i+", the label = "+str(label_number))
    for j in glob.iglob('dataset/'+i+'/*'):
        #samplerate, data = wavfile.read(j)
        y, s = librosa.load(j, sr=samplerate) # Downsample 44.1kHz
        #reduced_noise = nr.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)
        #print(s)
        #print(j)
        audio_data.append(y)
        #noise.append(y)
        labels.append(label_number)
        
    label_number = label_number + 1
#print(len(labels))
print(label_number)

# %%
# ANALIZANDO UNA MUESTRA 
example = 305
samples = audio_data[example]
sound_duration = (len(samples)/samplerate)
step = sound_duration/len(samples)
print("Duración: ", sound_duration,"s")
print("SampleTime: {:6f}s ".format(step))

# Visualizando Sonido en el Tiempo vs Amplitud
save_path= classes[labels[example]]+'.jpg'
print("Cantidad de muestras: ",len(samples))
plt.plot([step*i for i in range(len(samples))],samples)
plt.xlabel('Time s')
plt.ylabel('Amplitud []')
plt.show()
plt.title('signal in real time')
pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
#pylab.close()

# Analizando Sonido ZTiempo vs Coef STFT
time_init = time.time()
rcParams['figure.figsize'] = 20, 5
n_fft = 1024 # n_fft/2+1 como la cantidad de bandas a descomponer en el espectro de frecuencia
win_length=1024 # Ventaneo de la STFT
hop_length=int(win_length/2) # Desplazamiento de la ventana de transformación
freq = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data[example],n_fft=n_fft, win_length=win_length ,hop_length=hop_length)), ref=np.max) #Se obtiene la potencia de la transformada 
print("Dimención de la STFT: ", freq.shape)
print("Clase: ",classes[labels[example]])
print("Windowing Time: ", win_length*1/samplerate,"s")
time_final = time.time()
data = (time_final- time_init )/1000
print("tiempo de ejecución STFT: {:2f} uS".format(data*10**6))
save_path2 = 'SFTF_'+save_path
#plt.axis('off') # no axis
librosa.display.specshow(freq,hop_length=hop_length,x_axis='time', y_axis='linear',sr=44000)
plt.colorbar(format='%+2.0f dB')
plt.title('short time fourier transform')
pylab.savefig(save_path2, bbox_inches=None, pad_inches=0)
#pylab.close()
# %%
# Aplicando STFT a todas las muestras de sonido
X_stft = []
Y_label=[]
cant_samples = 220590
for i, i_sound in tqdm(enumerate(audio_data)):
    freq = librosa.amplitude_to_db(np.abs(librosa.stft(i_sound, n_fft=n_fft, win_length=win_length ,hop_length=hop_length)), ref=np.max)
    #freq = np.abs(librosa.stft(audio_data[i], n_fft=512, hop_length=256, win_length=512))
    freq=freq.reshape(-1,1)
    if freq.shape[0]==cant_samples:
        X_stft.append(freq)
        Y_label.append(labels[i])
    
X_stft =np.stack(X_stft) 
X_stft.shape
X_stft=X_stft.reshape(381,cant_samples)
X_stft.shape

# %% 
# Normalizando datos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_stft,y = X_stft)
normalized_stft = scaler.transform(X_stft)
print(np.amax(X_stft))
print(np.amax(normalized_stft))

# Dividiendo lasmuestras de forma homogenea
#from skmultilearn.model_selection import iterative_train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_stft,Y_label, test_size=0.20,random_state =2,stratify= Y_label)
print("Split ready")
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn import svm
from sklearn.decomposition import PCA

# %%
#Modelo con SVM
clf = svm.SVC(verbose= True,random_state=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM: acc:"+ str(accuracy_score(y_test, y_pred)))

#%%
# Modelo con PCA y SVM
print("Cantidad de caracteristicas", len(X_stft[0]))
pca = PCA(n_components=300)
pca.fit(normalized_stft) ## justa para todo el espectro de datos
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)
clf_PCA = svm.SVC(verbose= True,random_state=5, )
clf_PCA.fit(X_train_PCA, y_train)
y_pred = clf_PCA.predict(X_test_PCA)
print("SVM+PCA: acc:"+ str(accuracy_score(y_test, y_pred)))

#%% 
# Modelo PCA & SVM & GRIDsearch
#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
clf = svm.SVC(verbose= True,random_state=2)

param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
clf_svc = GridSearchCV(clf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = 4)
best_clf_svc = clf_svc.fit(X_train, y_train)
clf_performance(best_clf_svc,'SVC')


#%%
# Modelo clasificador con XGB & PCA
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

# %%
# Preparando datos para la CNN
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

model.add(Conv2D(32, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))#input_shape=features.shape[1:]
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(32))#input_shape=features.shape[1:]
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))
#sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4),
              metrics=['accuracy'])
model.summary()
n_epochs = 0
#tf.keras.utils.plot_model(model, to_file='NN_model.jpg', show_shapes=True)
# %% 
# Entrenando modelo Deep learning
logdir="logs2" 
epoch_add = 20
tboard_callback = TensorBoard(log_dir=logdir)
history = model.fit(X_train_NN, y_train_NN,
                    #steps_per_epoch = 8,   #cantidad de veces que se calculará el gradiente |DATOStotale = steps_per_epoch * batch_size
                    batch_size=8,          #cantidad de muestras para calcular el gradiente
                    epochs=n_epochs+epoch_add,
                    initial_epoch = n_epochs,
                    callbacks=[tboard_callback],
                    validation_data = (X_test_NN,y_test_NN))
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
