# building a small CNN to distinguish between cats and dogs

# download the dataset from https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip and put into utils folder

import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import layers
from keras import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

def unzip_directory(zip_file):

    zip_dir = zipfile.ZipFile(zip_file, mode='r')
    zip_dir.extractall('utils/')
    zip_dir.close()


def main():

    # unzipping the directory with the train and test data
    #unzip_directory('utils/cats_and_dogs_filtered.zip')

    #### inspecting the data ####
    base_dir = 'utils/cats_and_dogs_filtered'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    train_cats_filenames = os.listdir(train_cats_dir)
    train_dogs_filenames = os.listdir(train_dogs_dir)

    print('Number of training cat images: ', len(os.listdir(train_cats_dir)))
    print('Number of training dog images: ', len(os.listdir(train_dogs_dir)))
    print('Number of validation cat images: ', len(os.listdir(validation_cats_dir)))
    print('Number of validation dog images: ', len(os.listdir(validation_dogs_dir)))

    #### visualizing the data - we'll display 8 pictures at a time ####
    ncols, nrows = 4, 4
    fig = plt.gcf()  # get current figure if exists else create a new one
    fig.set_size_inches(nrows * 4, ncols * 4)  # view the images in a 4 x 4 configuration

    index = 0
    index += 8
    cat_pics_filepaths = [os.path.join(train_cats_dir, filename) for filename in train_cats_filenames[index-8:index]]
    dog_pics_filepaths = [os.path.join(train_dogs_dir, filename) for filename in train_dogs_filenames[index-8:index]]

    for i, img_path in enumerate(cat_pics_filepaths+dog_pics_filepaths):
        # setup subplot
        sub_plot = plt.subplot(nrows, ncols, i + 1)
        sub_plot.axis('off')
        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()

    #### building the classifier - model paramters have been chosen as they are a well known
    #### configuration and the small network will speed up training  ####

    img_input = layers.Input(shape=(150,150,3))

    # layer 1 - 16 lots of 3x3 convolutions followed by a 2x2 maxpooling layer
    x = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    # layer 2 - 32 lots of 3x3 convolutions followed by a 2x2 maxpooling layer
    x = layers.Conv2D(filters=32, kernel_size=3)(x)
    x = layers.MaxPooling2D(2)(x)

    # layer 3 - 64 lots of 3x3 convolutions followed by a 2x2 maxpooling layer
    x = layers.Conv2D(filters=64, kernel_size=3)(x)
    x = layers.MaxPooling2D(2)(x)

    # creating the fully connected layer - first need to flatten it into a 1-d tensor
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu')(x)

    x = layers.Dropout(rate=0.5) # an optional layer to help prevent overfitting

    output = layers.Dense(units=1, activation='sigmoid')(x)

    # create the model
    model = Model(img_input, output)

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

    #### DATA GENERATORS - Read in the pictures with the labels and feed them to the network ####
    # to make the image pixel values between [0, 1]
    # We artifically augment the training data to increase number of training samples.
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=(150,150),
                                                        batch_size=20,
                                                        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(directory=validation_dir,
                                                            target_size=(150, 150),
                                                            batch_size=20,
                                                            class_mode='binary')

    #### MODEL TRAINING ####
    training = model.fit_generator(generator=train_generator,
                                   steps_per_epoch=100, # 2000 training images and a batch size of 20)
                                   epochs=15,
                                   validation_data=validation_generator,
                                   validation_steps=50, # 1000 validation images and a batch size of 20
                                   verbose=2)

    acc = training.history['acc']
    val_acc = training.history['val_acc']

    loss = training.history['loss']
    val_loss = training.history['val_loss']

    epochs = range(len(acc))

    # plotting training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title("Training and validation accuracy")
    plt.figure()

    # plotting training and validation loss per epoch
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.figure()


if __name__ == '__main__': main()

