# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:07:46 2021
Note: Message and Block is used interchangeably. A message contains M number of 'symbols'.
@author: aguboshimec
"""

#General imports
import numpy as np
from numpy import arange, argmax, array
import matplotlib.pyplot as plt
import keras
import copy
from keras import backend as K
from keras import layers, regularizers
from keras.models import Sequential, Model
from keras.layers import Dense,Input, UpSampling1D, Dropout, MaxPooling1D, Activation, GaussianNoise,BatchNormalization,  Flatten, ZeroPadding2D, Conv1D, Conv2D, MaxPooling2D, Lambda, Layer
from keras.utils import plot_model, to_categorical
from tensorflow import keras


#parameter and variable declaration:
k = 2          # information bits per symbol eg. QPSK would have 2bitperSymbol.
n = 2          # channel use per message. I'd liken a message to number of symbols of a constellation
M = 2**k         # messages could reflect the modulation order or number of symbols.
R = k/n        # effective throughput or communication rate or bits per symbol per channel use

EbNo_dB = -12    # Eb/N0 used for training. #to make model more robust
noise_stdDev = np.sqrt(1 / (2*R*10**(EbNo_dB/10))) # Noise Standard Deviation
noOfSamples = int(2000) #number of sample data generated in the for-loop. It determines the 3 dimension of our matrix or vector.
epoch = 50
batchSize = 200
inputsize = M*M  #since I was considering a dense model. I kinda flattened
data_all = None

#Generate training dataset
def dataset():
    global data_all
    data_all = []
    for i in range (noOfSamples):
        data = np.random.randint(low=0, high=M, size=(M, ))
        # the convert to one-hot encoded version
       
        data = to_categorical(data, num_classes= M)
        data_all.append(data)        
dataset() 

data_all = array(data_all) #convert to an array or 3D tensor   

#data set formatting. equally Splits dataset into training and validation:
data_all = np.array_split(data_all, 2)
x__train = data_all[1] #training
x__validtn = data_all[0] #validation
x_train = np.reshape(data_all[1], (len(x__train),inputsize)) #training
x_validtn = np.reshape(data_all[0], (len(x__validtn),inputsize)) #validation

#AutoEncoder with Conv1D Layer
# Define Power Norm for Tx
def normalization(x):
    mean = K.mean(x ** 2)
    return x / K.sqrt(2 * mean)  # 2 = I and Q channels

# Define Channel Layers including AWGN and Flat Rayleigh fading
#  x: input data. sigma: noise std
def channel_layer(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    return x + w

model_symb = Input(batch_shape=(batchSize, M, M), name='input_symbls')

#Encoder
x = Conv1D(filters= 64, strides=1, kernel_size=1, name='e_1')(model_symb)
x = BatchNormalization(name='e_2')(x)
x = Activation('relu', name='e_3')(x)

x = Conv1D(filters=64, strides=1, kernel_size=1, name='e_4')(x)
x = BatchNormalization(name='e_5')(x)
x = Activation('relu', name='e_6')(x)

x = Conv1D(filters= 2, strides=1, kernel_size=1, name='e_7')(x)  # 2 = I and Q channels
x = BatchNormalization(name='e_8')(x)
x = Activation('relu', name='e_9')(x)
x = MaxPooling1D(2)(x)


# AWGN channel
y_h = Lambda(channel_layer, arguments={'sigma': noise_stdDev}, name='channel_layer')(x)

# Define Decoder Layers (Receiver)
y = UpSampling1D(2)(y_h)
y = Conv1D(filters=64, strides=1, kernel_size=1, name='d_1')(y)
y = BatchNormalization(name='d_2')(y)
y = Activation('relu', name='d_3')(y)

y = Conv1D(filters=64, strides=1, kernel_size=1, name='d_7')(y)
y = BatchNormalization(name='d_8')(y)
y = Activation('relu', name='d_9')(y)

# Output One hot vector and use Softmax to soft decoding
model_output = Conv1D(filters= M, strides=1, kernel_size=1, name='d_10', activation='softmax')(y)

# Build System Model
sys_model = Model(model_symb, model_output)
encoder = Model(model_symb, x)

# Print Model Architecture
sys_model.summary()

sys_model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#display the input and output shapes of each layer:
plot_model(sys_model, "autoencoder.png", show_shapes=True)

autoencoderConv1D = sys_model.fit(x__train, x__train, validation_data = (x__validtn, x__validtn), epochs=epoch, shuffle = True, batch_size =  batchSize, verbose= 1)


def test_dataset1D():
    global test_data1D_all, test_data1D, test__data_all
    test_data1D_all = []
    test__data1D_all = []
    for i in range (1000): #generates xx samples
        test_data1D = np.random.randint(low=0, high=M, size=(M, ))
        test__data1D = copy.copy(test_data1D)
        # the convert to one-hot encoded version
        test_data1D = to_categorical(test_data1D, num_classes= M)
        test__data1D_all.append(test__data1D) #copy of the orignal test data
        test_data1D_all.append(test_data1D)        
test_dataset1D()

# used as test data
test_data1D_all = array(test_data1D_all) #convert to an array or 3D tensor   

error_rate_all = []
EbNo_dB_all = []
for EbNo_linear in arange(1,20):
    x_test_noisy = (1/0.2) + test_data1D_all #adds noise
    #encoded_data = encoder.predict(xxxxx) #for now, I dont need to show the encoded data or message
    decoded_data1D = sys_model.predict(x_test_noisy)
    
    position = np.argmax(decoded_data1D, axis=2)
    test__data_all = array(test__data_all)
    x_test_predicted = np.reshape(position, newshape = test__data_all.shape) 
    error_rate = np.mean(np.not_equal(test__data_all,x_test_predicted)) #compares, avergaes, and determines how many errors for each block
    
    error_rate_all.append(error_rate)
    EbNo_dB_all.append(10*np.log10(EbNo_linear)) #converts to dB, and the appends ready for plotting


#plots loss (b oth training and validation) over epoch:
#Error_Rate vs. EbNodB
plt.plot(EbNo_dB_all, error_rate_all)
plt.ylabel('block error rate')
plt.xlabel('EbNo (dB)')
plt.title('Error Rate vs. Eb/No')
plt.grid(b=None, which='major', axis='both')
plt.show()
    
#Loss
plt.plot(DeepNNConv2D.history['loss'])
plt.plot(DeepNNConv2D.history['val_loss'])
plt.title('Graph of final Performance Training Loss')
plt.ylabel('Loss')
plt.xlabel('No. of Epoch')
plt.legend(['Loss', 'Validation Loss'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()

#Accuracy
plt.plot(DeepNNConv2D.history['accuracy'])
plt.plot(DeepNNConv2D.history['val_accuracy'])
plt.title('Graph of Performance Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epoch')
plt.legend(['Accuracy', 'Validation Accuracy'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()




