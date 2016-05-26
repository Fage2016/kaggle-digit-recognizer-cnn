import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils

# The competition datafiles are in the same directory
# Read competition data files:
train = pd.read_csv("./train.csv").values
test = pd.read_csv("./test.csv").values

epoch = 50

batch_size = 128
img_rows, img_cols = 28, 28

filters = [64, 128]
kernel = 3
pool = 2

trainX = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0

trainY = kutils.to_categorical(train[:, 0])
nb_classes = trainY.shape[1]

cnn = models.Sequential()

cnn.add(conv.ZeroPadding2D((1, 1), input_shape=(1, img_rows, img_cols),))

cnn.add(conv.Convolution2D(filters[0], kernel, kernel))
cnn.add(core.Activation('relu'))
cnn.add(conv.MaxPooling2D(strides=(pool, pool)))

cnn.add(conv.ZeroPadding2D((1, 1)))

cnn.add(conv.Convolution2D(filters[1], kernel, kernel))
cnn.add(core.Activation('relu'))
cnn.add(conv.MaxPooling2D(strides=(pool, pool)))

cnn.add(conv.ZeroPadding2D((1, 1)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.5))
cnn.add(core.Dense(128))
cnn.add(core.Activation('relu'))
cnn.add(core.Dense(nb_classes))
cnn.add(core.Activation('softmax'))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=epoch, verbose=1)

testX = test.reshape(test.shape[0], 1, img_rows, img_cols)
testX = testX.astype(float)
testX /= 255.0

yPred = cnn.predict_classes(testX)

np.savetxt('mnist.csv', np.c_[range(1, len(yPred)+1), yPred], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
