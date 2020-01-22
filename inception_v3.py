# Building a cat/dog classifier using a pretrained model (Google's inception v3)
# We still use the same subset of cats/dogs dataset in the utils folder.
# download the weights from (using curl in terminal):
# https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5


import os
from keras import layers
from keras import Model
from keras_applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights_file = 'utils/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# include_top=False means we don't want the final fully connected layer
pre_trained_model = InceptionV3(include_top=False, input_shape=(150, 150, 3), weights=None)
pre_trained_model.load_weights(weights_file)

# we don't want the model to be trainable since we're only using it for feature extraction
for layer in pre_trained_model.layers:
    layer.trainable = False

# Let's use the 'mixed7' layer as the input to our model
last_layer = pre_trained_model.get_layer('mixed7')
print("last later output shape: ", last_layer.output_shape)
last_output = last_layer.output # this is the input to our own model


# building our own model to on top of last_layer
x = layers.Flatten()(last_output) # flattening output layer to 1-dim
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(rate=0.2)(x)
x = layers.Dense(units=1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc']
              )

# now for the data

base_dir = 'utils/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cats_filenames = os.listdir(train_cats_dir)
train_dogs_filenames = os.listdir(train_dogs_dir)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(directory=validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

training = model.fit_generator(generator=train_generator,
                               steps_per_epoch=100, # 2000 training images and a batch size of 20)
                               epochs=2,
                               validation_data=validation_generator,
                               validation_steps=50, # 1000 validation images and a batch size of 20
                               verbose=2)

# This is a big improvement but we can improve even more by fine-tuning the final layers of the model.

from keras.optimizers import SGD

unfreeze = False

for layer in pre_trained_model.layers:
    if unfreeze:
        layer.trainable = True
    if layer.name == 'mixed6':
        unfreeze = True

model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['acc'])

training = model.fit_generator(generator=train_generator,
                               steps_per_epoch=100,
                               epochs=50,
                               validation_data=validation_generator,
                               validation_steps=50,
                               verbose=2)

# plotting it

acc = training.history['acc']
val_acc = training.history['val_acc']

loss = training.history['loss']
val_loss = training.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title("Training and Validation Accuracy")
plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title("Training and Validation Loss")



