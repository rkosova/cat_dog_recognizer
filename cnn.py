from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initializing CNN as a sequence of layers
classifier = Sequential()

#Step 1: Convolution
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation = "relu"))
#32 is common practice for the start n of filters
#input shape takes number of channels (colors), and shape of images
#reLU is used to remove nonlinearity by not allowing non zero pixel values
classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), activation = "relu"))


#Step 2: Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#we use pooling to lower the number of input nodes after flattening
#num of strides is the stride is equal to pool_size

#Step 3: Flattening 
classifier.add(Flatten())
#Flattening is done in conjunction with convolution and pooling
#in order to get the spatial relationship between the pixels
#that's why we don't flatten an image at first
#we convolve it and pool it to extract the features and then we flatten it

#Step 4: Full conection
classifier.add(Dense(units=128, activation = "relu"))
classifier.add(Dense(units=1, activation = "sigmoid"))
#structuring the aritificial neural network

#Step 5: Compilation
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Step 6: Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data_cnn/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('data_cnn/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=1000,
                         epochs=15,
                         validation_data=test_set,
                         validation_steps=2000)

#pre made preprocessing template
#used to minimalize overfitting on training set
#by augmenting the data in different ways

classifier.save("models/dog_cat_classifier2.h5")
print("Model saved to disk.")
