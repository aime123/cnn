#La preparation des donnees est tres importante de meme que la mise sous meme echelle des donnees

#Partie 1 - Construction du CNN
#Importation des modules

from keras.models import Sequential 
#permet sert a initialiser notre reseau de neurones
#En realite, il y a 2 manieres d'initialiser le reseau de eurones de manieres sequentielles ou avec un graphe

from keras.layers import Convolution2D
#Comme ce sont des photos 
#Si c'etait en 3D on ajouterait une 3eme D avec la variable temps

from keras.layers import MaxPooling2D

from keras.layers import Flatten #pour applatir les donnees avant de les rentrer dans le reseau de neurones

from keras.layers import Dense #pour creer les couches completement connecte...

#Initialiser le reseau de neurones
#On est toujours de faire une classification

classifier = Sequential()

#Donc au debut on convertit notre image en une matrice avec des nombres pour chaque pixel
 
#Etape1: Convolution
classifier.add(Convolution2D(filters=32,kernel_size=[3,3], strides= 1, input_shape = (128,128,1), activation= "relu"))

#Le input_shape = (64,64,1) important au debut car le resau nesait pas ce qu'il va prendre en entree
#kernel_size permet d'avoir la taille du feature dectector qui pouvait aussi etre 5x5 ou meme 7x7
#strides permet de savoir le deplacement du feature dectector
#input_shape permet d'avoir la forme de l'image. Ici notre image est de taille 64pixels x 64pixels
# Notons que pour aller plus vite d'autant plus que nous travaillons avec du cpu il faut prendre 64x64
# Lorsque nous voudrons travailler avec du GPU on pourra prendre les 128x128 ou plus 


#Etape2: Max Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

#1prime: Rajout d'une couche de convolution pour ameliorer le modele
#Le input_shape = (64,64,1) n'est plus necessaire car le reseau de neurones connait deja la structure du modele
classifier.add(Convolution2D(filters=32,kernel_size=[3,3], strides= 1,  activation= "relu"))

#Etant donne que chaque couche de convolution est suivie d'une couche de pooling alors:
classifier.add(MaxPooling2D(pool_size= (2,2)))


#Pour ameliorer le modele il y a plusieurs autres manieres
#Augmenter la taille des images, les pixels
#Ajouter d'autres couches cachees
#Ajouter 

#Etape3: Flattening
classifier.add(Flatten())


#Etape4: Couche entierement connectee
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))#VIP Parce qu'on a une seule sortie, on utilise la fonction sigmoid
#Si on avait plusieurs sorties on aurait utilise.. la fonction softmax

#Compilation
#Choisir l'algo de gradient, la fonction de cout et une matrice pour mesurer la performance de notre reseau de neurnes

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Entrainer le CNN sur nos images
#On va utiliser un procede qui s'appelle "image augmentation" en anglais
#Preparer toutes nos images pour eviter le surentraiement
#Si on saute cette etape notre entrainement peut avoir de bons resultats sur le jeux de texte mais bizarre sur le jeux d'entrainement

# prepare data augmentation configuration
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset\\training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale')

test_set = test_datagen.flow_from_directory(
        'dataset\\test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale')

# fine-tune the model
classifier.fit(
    training_set,
    steps_per_epoch=88,
    epochs=25,
    validation_data=test_set,
    validation_steps=9
)

# Après l'entraînement, sauvegardez le modèle
classifier.save('cnn_binaire.h5')

'''#Prediction:
import numpy as np
from tensorflow.keras.preprocessing import image

# Modification du chargement de l'image de test pour utiliser le mode niveaux de gris
test_image = image.load_img("img.jpg", 
                          target_size=(128, 128),
                          color_mode='grayscale')  # Ajouter color_mode='grayscale'
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Normalisation comme dans le générateur
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    print("Cerveau sain")
else:
    print("Cerveau malade")'''
    


