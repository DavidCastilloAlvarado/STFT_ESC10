from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pylab
from sklearn.model_selection import train_test_split, GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score , plot_confusion_matrix,confusion_matrix
import itertools
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier as KNC
import tensorflow.keras as keras
import librosa    
import glob
from pylab import rcParams
import random
import time
from tqdm import tqdm
import IPython.display as ipd


def readsounfiles_aumenteddata(folder = 'dataset/', 
                               samplerate = 44000, 
                               sound_duration = 4,
                               mult_generaciones = 5,
                               noise_var = 0.0025,
                               random_state=10):
    audio_data = []
    classes = []
    label_number=0
    labels = []
    sound_duration = 4 # se desea que todas las muestren duren 4 segundos y se las fuerza a serlo
    max_leng = int(sound_duration*samplerate)
    break_points     = [i for i in range(0,max_leng,int(max_leng/int(sound_duration*2)) )] # posición en tiempo discreto
    break_points[-1] = max_leng
    noise_list = [np.stack([random.uniform(-noise_var,noise_var) for _ in range(max_leng)]), 
                  np.stack([random.gauss(0,noise_var) for _ in range(max_leng)]),
                  np.stack([0 for _ in range(max_leng)])]

    def swap_audio_seconds(audio, audio_2, break_points, random_state = 2, half = True, rand_gain=10, noise_list=None):
        random.seed(random_state)
        max_leng = break_points[-1]
        positions = [i for i in range(0,len(break_points)-1)]
        random.shuffle(positions)
        noise =noise_list[random.randint(0,2)]
        #print(positions)
        def append_sound_parts(audio,audio_2,break_points,positions,ind):
            sound_temp = []
            if ind == len(positions)-1:
                return audio[break_points[positions[ind]]:break_points[positions[ind]+1]]
            elif ind%2 == 0:
                peace2add = audio_2[break_points[positions[ind]]:break_points[positions[ind]+1]]
                return np.append( peace2add if np.max(np.abs(peace2add)) >= 0.2 else audio[break_points[positions[ind]]:break_points[positions[ind]+1]] ,append_sound_parts(audio,audio_2,break_points,positions,ind+1) )
            else:
                return np.append(audio[break_points[positions[ind]]:break_points[positions[ind]+1]],append_sound_parts(audio,audio_2,break_points,positions,ind+1) )
        def swap_half_position(max_leng,audio):
            return np.append(audio[int(max_leng/2):], audio[:int(max_leng/2)]) 

        return swap_half_position(max_leng,audio)+noise if half else (append_sound_parts(audio,audio_2,break_points,positions,0)+noise)*np.power(10, random.uniform(-rand_gain, rand_gain) / 20.0)

    for filepath in glob.iglob('dataset/*'):
        #print(filepath[9:])
        #print(filepath)
        classes.append(filepath[8:])
    print(classes)


    for i in classes:
        print("the class = "+i+", the label = "+str(label_number))
        for ind, sound_file in tqdm( enumerate(glob.iglob(folder+i+'/*'))):
            #samplerate, data = wavfile.read(j)
            #print(sound_file)

            i_sound, s = librosa.load(sound_file, sr=samplerate) # Downsample 44.1kHz
            i_sound = i_sound[:max_leng] if len(i_sound) > max_leng else np.pad(i_sound, (0,max_leng-len(i_sound)), constant_values = (0,0)) #padding or cut
            audio_2 = audio_data[-(mult_generaciones+1)] if ind > 0 else swap_audio_seconds(i_sound,i_sound,break_points,noise_list=noise_list)
            audio_data.append(i_sound)
            _= [audio_data.append(swap_audio_seconds(i_sound,audio_2,break_points,half=False, random_state=(_seed+(label_number+1)*ind), noise_list=noise_list )) for _seed in range(random_state,random_state+mult_generaciones)]
            _= [labels.append(label_number) for _ in range(0,mult_generaciones+1)]
            #samplesize = len(y)
            #reduced_noise = nr.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)
            #print(s)
            #print(j)
            #audio_data.append(i_sound)
            #max_leng = samplesize if samplesize>max_leng else max_leng
            #noise.append(y)
            #labels.append(label_number)
        label_number = label_number + 1
        print("Cantidad máxima de muestras / Sonido / clase: ", max_leng)
    return np.stack(audio_data), labels , classes

def STFT_sound_DB(audio_data,labels,n_fft,win_length,hop_length):
    X_stft = []
    Y_label=[]
    for i, i_sound in tqdm(enumerate(audio_data)):
        #i_sound = i_sound[:cant_samples] if len(i_sound) > cant_samples else np.pad(i_sound, (0,cant_samples-len(i_sound)), constant_values = (0,0))
        coef_stft = librosa.amplitude_to_db(np.abs(librosa.stft(i_sound, n_fft=n_fft, win_length=win_length ,hop_length=hop_length)), ref=np.max)
        coef_stft = np.transpose(coef_stft)
        stft_shape = coef_stft.shape
        #freq = np.abs(librosa.stft(audio_data[i], n_fft=512, hop_length=256, win_length=512))
        X_stft.append(coef_stft.reshape(-1,1))
        Y_label.append(labels[i])
#         audio_data[i]=None
#         labels[i] = None
        # if freq.shape[0]>=cant_samples:
        #     X_stft.append(freq[:cant_samples])
        #     Y_label.append(labels[i])
    return np.stack(X_stft).reshape(len(audio_data),-1) , Y_label , stft_shape

# Auxiliar FUNC
    
def plot_confusion_matrix(y_true, y_pred, class_names,title="Confusion matrix",normalize=False,onehot = False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    if onehot :
        cm = confusion_matrix([y_i.argmax() for y_i in y_true], [y_ip.argmax() for y_ip in y_pred])
    else:
        cm = confusion_matrix(y_true, y_pred)
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2) if normalize else cm

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #return figure