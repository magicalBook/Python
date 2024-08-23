# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:50:43 2021

@author: CHENTIEJUN
"""

from keras.models import *
from keras.layers import *
from keras import optimizers
from keras import backend as K
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Permute
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


train_dir = input('entry train dir: ')
validation_dir = input('entry validation dir: ')
epoch_input = int(input('entry epochs: '))

batch_size = 128

train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                        train_dir,
                        target_size=(150,200),
                        batch_size=batch_size)


validation_generator = test_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(150,200),
                        batch_size=batch_size)


input_tensor = Input((150, 200, 3))

incep = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 200, 3))

for layer in incep.layers:
    layer.trainable = False
    

cnn_model = Model(inputs=incep.input, outputs=incep.output)
x = cnn_model(input_tensor)

x = Reshape((24, 4, -1))(x)
#x = Permute((2,1,3))(x)
x = TimeDistributed(Flatten())(x)

lstm = LSTM(256)(x)
x = Dropout(0.25)(lstm)
x = Dense(2, activation='softmax')(x)

base_model = Model(input_tensor, x)

steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

base_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.Adam(lr=2e-4),
                   metrics=['acc'])



print(base_model.summary())

history = base_model.fit_generator(
                        train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epoch_input,
                        validation_data=validation_generator,
                        validation_steps=validation_steps)



train_acc = history.history['acc']
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epoch = range(1,len(train_acc)+1)

plt.plot(epoch,train_acc,'r',label='train accuracy')
plt.plot(epoch,val_acc,'g',label='validation accuracy')
plt.title('train and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epoch,train_loss,'r',label='train loss')
plt.plot(epoch,val_loss,'g',label='validation loss')
plt.title('train nad validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
