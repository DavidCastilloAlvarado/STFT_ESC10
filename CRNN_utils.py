import numpy as np
import librosa  
import pylab
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,LSTM,Reshape,Input, Lambda,Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow.keras as keras
import random
import time

class DataSoundGenerator(tf.keras.utils.Sequence):
#     samplexClass     = self.Yorig.count(self.Yorig[0])
    
    def __init__(self, Xsound, Yclass, batch_size,n_fft,win_length,hop_length,
                 childrens=2,samplerate=44000, 
                 num_classes=None, shuffle=True,
                 noise_var=0.0025,rand_gain_db=10, 
                 random_state=None):
        assert batch_size%(childrens+1) == 0 , 'batch_size debe ser multiplo de (childrens+1)'
        self.batch_sizes = batch_size//(childrens+1)
        self.childrens = childrens
        self.rand_gain = rand_gain_db
        self.samplerate = samplerate
        self.n_fft      = n_fft
        self.win_length = win_length 
        self.hop_length = hop_length
        self.Xorig   = Xsound
        self.Yorig   = Yclass
        self.random_state = random_state
        self.indices = [i for i in range(len(self.Xorig))]
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.max_leng         = len(self.Xorig[0])
        self.sound_duration   = self.max_leng/samplerate 
        assert len(Xsound[0])>self.sound_duration*2 >= 0 , 'La cantidad de muestras tiene que ser mayor al valor numérico de dos veces el tiempo de duración en segundos'
        self.break_points     = [i for i in range(0,self.max_leng,int(self.max_leng/int(self.sound_duration*2)) )] # posición en tiempo discreto
        self.break_points[-1] = self.max_leng
        self.noise_list = [np.stack([random.uniform(-noise_var,noise_var) for _ in range(self.max_leng)]), 
                              np.stack([random.gauss(0,noise_var) for _ in range(self.max_leng)]),
                              np.stack([0 for _ in range(self.max_leng)])]
        self.on_epoch_end()
        self.__info()
        self.shape_spectro = self.audio_to_spectrogram(audio=Xsound[0]).shape
        
    def __info(self):
        print("Generador de Sonido =====>")
        print("Se cuenta con {:4d} datos originales".format(len(self.indices)))
        print("Se generarán  {:4d} datos sintéticos más".format(len(self.indices)*self.childrens))
        
    def __len__(self):
        #batches per epochs
        return len(self.indices) // self.batch_sizes

    def __getitem__(self, index):
        index = self.index[index * self.batch_sizes:(index + 1) * self.batch_sizes]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
    
    def __get_data(self, batch):
        X = np.empty((self.batch_sizes*(self.childrens+1),self.shape_spectro[0],self.shape_spectro[1]))
        y = np.empty((self.batch_sizes*(self.childrens+1), self.num_classes), dtype=int)
        #print(batch)
        for i, id in enumerate(batch):
            y_yi = self.Yorig[id]
            hotencode_yi = keras.utils.to_categorical(y_yi, num_classes=self.num_classes, dtype='float32')
            X[i*(self.childrens+1)] = self.audio_to_spectrogram(self.Xorig[id]) # original X
            y[i*(self.childrens+1)] = hotencode_yi # original Y
            for i_aum in range(1,self.childrens+1):
                random_base = (self.random_state+1)*id + y_yi+i_aum
                random.seed(random_base)
                audio0 = self.Xorig[random.choice( np.where(self.Yorig==y_yi)[0]) ] # se elige aleatoriamente otra muestra de la misma categoría para fusionarla con la original
                random.seed(random_base+1)
                audio1 = self.Xorig[random.choice( np.where(self.Yorig==y_yi)[0]) ] # se elige aleatoriamente otra muestra de la misma categoría para fusionarla con la original
                audio2 = self.swap_audio_seconds(audio=audio0,audio_2= audio1, random_state = random_base+ random.choice( np.where(self.Yorig==y_yi)[0]) )
                X[i*(self.childrens+1)+i_aum] = self.audio_to_spectrogram(self.swap_audio_seconds(audio=self.Xorig[id],audio_2= audio2, random_state = random_base+ random.choice( np.where(self.Yorig==y_yi)[0])+1)) #self.Xorig[id]*(i_aum+0.01) # generado X
                y[i*(self.childrens+1)+i_aum] = hotencode_yi # self.Yorig[id]# generado Y ##
        return np.reshape(X,(self.batch_sizes*(self.childrens+1),self.shape_spectro[0], -1,1)), y
    
    def swap_audio_seconds(self, audio, audio_2, random_state = None):
        random.seed(random_state)
        max_leng = self.break_points[-1]
        positions = [i for i in range(0,len(self.break_points)-1)]
        random.shuffle(positions)
        noise = self.noise_list[random.randint(0,len(self.noise_list)-1)]
        #print(positions)
        def append_sound_parts(audio,audio_2,positions,ind):
            sound_temp = []
            if ind == len(positions)-1:
                return audio[self.break_points[positions[ind]]:self.break_points[positions[ind]+1]]
            elif ind%2 == 0:
                peace2add = audio_2[self.break_points[positions[ind]]:self.break_points[positions[ind]+1]]
                #print(audio_2.shape)
                return np.append( peace2add if np.max(np.abs(peace2add)) >= 0.2 else audio[self.break_points[positions[ind]]:self.break_points[positions[ind]+1]] ,append_sound_parts(audio,audio_2,positions,ind+1) )
            else:
                return np.append(audio[self.break_points[positions[ind]]:self.break_points[positions[ind]+1]],append_sound_parts(audio,audio_2,positions,ind+1) )
        return (append_sound_parts(audio,audio_2,positions,0)+noise)*np.power(10, random.uniform(-self.rand_gain, self.rand_gain) / 20.0)
    
    def audio_to_spectrogram(self, audio):
        coef_stft = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=self.n_fft, win_length=self.win_length ,hop_length=self.hop_length)), ref=np.max)
        coef_stft = np.transpose(coef_stft)
        min_val = np.min(coef_stft)
        return (coef_stft - min_val)/abs(min_val) # Normalizando



def model_CRNN(input_shape_nn,reshape_time_len,reshape_feature_len,n_clases=10, saved_file = None):
    initializer = tf.random_normal_initializer(0,0.02)
    model = Sequential()
    model.add(Input(input_shape_nn ))

    model.add(Conv2D(8, (5, 5),padding='same', kernel_initializer = initializer ) )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5),padding='same', kernel_initializer = initializer,use_bias = False ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Reshape((reshape_time_len, 16*reshape_feature_len)))
    #model.add(Permute((2, 1)))
    model.add(Bidirectional(LSTM(32,return_sequences=True)))
    model.add(LSTM(32,return_sequences=False,dropout=0.2 ))
    #model.add(Flatten())
    model.add(Dense(16, kernel_initializer = initializer))#input_shape=features.shape[1:]
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(n_clases,kernel_initializer = initializer))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    #sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    #tf.keras.utils.plot_model(model, to_file='NN_model.jpg', show_shapes=True)
    if (saved_file):
        model.load_weights(saved_file)
        try:
            #model.load_model(saved_file)
            print("Pesos cargados")
        except:
            print("No se puede cargar los pesos")
    model.summary()
    return model