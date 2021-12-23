import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

train_batch_size = 500
img_height = 512
img_width = 382

# ------------------- For the training set -------------------
train_hd_ds = tf.keras.preprocessing.image_dataset_from_directory('./archive/train', image_size=(img_height, img_width), batch_size=train_batch_size)
train_hd_ds = train_hd_ds.cache().prefetch(buffer_size=train_batch_size)


#Create the labels, 0 means it's a hotdog, 1 is hotdog 
train_labels = []
for i in range(250):
    train_labels.append(0)

for i in range(250):
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


# ------------------- Creating new images from the data set by performing transformations -------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 10,
    fill_mode = 'nearest',
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip = True,
    brightness_range = [0.4, 1.5, 2.6],
    zoom_range=0.2
)

quit()

newImages = datagen.flow(train_ds, batch_size=500)

train_ds = train_ds + newImages

train_batch_size = len(train_ds)

# ------------------- Normalizing the data -------------------
datagen = ensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1.0/255.0,
    featurewise_center = True,
    featureWise_std_normalization = True
)

train_iterator = datagen(train_ds, train_labels, batchSize = train_batch_size)
test_iterator = datagen(test_ds, test_labels, batchSize = 500)


# ------------------- Trainging the model -------------------
model = keras.sequential([
    layers.Dense(2, activation="relu", name="l1"),
    layers.Dense(3, activation="relu", name="l2"),
    layers.Dense(4, name="l3"),
])

#An Creating an early stopping variable 
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', patience = 5, restore_best_weights = True)

history = model.fit(train_iterator, batch_size = train_batch_size, Epochs = 50, validation= (test_iterator, test_labels), callbacks = [earlyStopping])

