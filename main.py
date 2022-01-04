import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_batch_size = 498
img_height = 512
img_width = 382

# ------------------- For the training set -------------------

#train_hd_ds = tf.keras.preprocessing.image_dataset_from_directory(
#    "archive/train", 
#    labels='inferred',
#    image_size=(img_height, img_width), 
#    batch_size=train_batch_size
#)

train_path = 'archive/train'

train_hd_ds = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,classes=['hotdog', 'not_hotdog'], batch_size=10)

print(train_hd_ds)

print(len(list(train_hd_ds)))
quit()

train_hd_ds = train_hd_ds.cache().prefetch(buffer_size=train_batch_size)

#Create the labels, 0 means it's a hotdog, 1 is hotdog 
train_labels = []
for i in range(249):    #It's missing an image in each set so I account for that in the label creation
    train_labels.append(0)

for i in range(249):
    train_labels.append(1)

#Amalgomate them into one trainging set
train_ds = train_hd_ds


# ------------------- For the Testing set -------------------
test_hd_ds = tf.keras.preprocessing.image_dataset_from_directory('./archive/test', image_size=(img_height, img_width), batch_size=train_batch_size)
test_hd_ds = test_hd_ds.cache().prefetch(buffer_size=train_batch_size)


#Create the labels, 0 means it's a hotdog, 1 is hotdog 
test_labels = []
for i in range(250):
    test_labels.append(0)

for i in range(250):
    test_labels.append(1)

#Amalgomate them into one testing set
test_ds = test_hd_ds


# ------------------- Creating new images from the data set by performing transformations as well as normalize all the data-------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 10,
    fill_mode = 'nearest',
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [0.4, 2.6],
    zoom_range = 0.2,
    rescale = 1.0/255.0,
    featurewise_center = True,
    featurewise_std_normalization = True
)

print(train_ds)

train_ds = np.array(train_ds)
train_labels = np.array(train_labels)

print(len(train_ds))
print(len(train_labels))

newImages = datagen.flow(train_ds, train_labels, batch_size = train_batch_size)

#Since imageDataGenerator replaces data, I want to add to it so I am adding it to the original dataset
train_ds = train_ds + newImages 

train_batch_size = len(train_ds)

# ------------------- performing it on the test data -------------------
test_iterator = datagen(test_ds, test_labels, batchSize = 500)

quit()

# ------------------- Trainging the model -------------------
model = keras.sequential([
    layers.Dense(2, activation="relu", name="l1"),
    layers.Dense(3, activation="relu", name="l2"),
    layers.Dense(4, name="l3"),
])

#An Creating an early stopping variable 
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', patience = 5, restore_best_weights = True)

history = model.fit(train_iterator, batch_size = train_batch_size, Epochs = 50, validation= (test_iterator, test_labels), callbacks = [earlyStopping])

