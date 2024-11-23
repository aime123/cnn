# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:40:44 2024

@author: PC
"""

# Importation des modules
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialiser le réseau de neurones
classifier = Sequential()

# Étape 1 : Convolution
classifier.add(Convolution2D(filters=32, kernel_size=[3,3], strides=1, input_shape=(128, 128, 1), activation="relu"))

# Étape 2 : Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Ajout d'une couche de convolution supplémentaire
classifier.add(Convolution2D(filters=32, kernel_size=[3,3], strides=1, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Étape 3 : Flattening
classifier.add(Flatten())

# Étape 4 : Couche entièrement connectée
classifier.add(Dense(units=128, activation="relu"))

# Couche de sortie avec 4 neurones (pour 4 classes) et fonction d'activation softmax
classifier.add(Dense(units=4, activation="softmax"))

# Compilation du modèle
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Préparation de la configuration de data augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des datasets
training_set = train_datagen.flow_from_directory(
        'dataset\\training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',  # Pour plusieurs classes
        color_mode='grayscale')

test_set = test_datagen.flow_from_directory(
        'dataset\\test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',  # Pour plusieurs classes
        color_mode='grayscale')

# Entraînement du modèle
classifier.fit(
        training_set,
        steps_per_epoch=88,
        epochs=25,
        validation_data=test_set,
        validation_steps=9)
