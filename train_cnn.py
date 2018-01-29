from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, add, concatenate


# load the mnist data
(x_train, y_train), (x_val, y_val) = mnist.load_data()


# reshape the data in three channels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

# pixel values are originally ranged from 0-255, rescale it to 0-1
x_train = x_train.astype('float32')/255.0
x_val = x_val.astype('float32')/255.0

# one-hot output vector 
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)


inputs = Input(shape=(28, 28, 1), dtype='float32')

# a two layer deep cnn network 
# 64 and 128 filters with filter size 3*3
# max-pool size 2*2 - it will downscale both the input dimensions into halve 

conv_a1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
max_pool_a1 = MaxPooling2D(pool_size=(2, 2))(conv_a1)

conv_a2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool_a1)
max_pool_a2 = MaxPooling2D(pool_size=(2, 2))(conv_a2)
out_a = Flatten()(max_pool_a2)


# another two layer deep cnn network 
# 64 and 128 filters with filter size 4*4
# max-pool size 2*2 - it will downscale both the input dimensions into halve 

conv_b1 = Conv2D(64, kernel_size=(4, 4), activation='relu', padding='same')(inputs)
max_pool_b1 = MaxPooling2D(pool_size=(2, 2))(conv_b1)

conv_b2 = Conv2D(128, kernel_size=(4, 4), activation='relu', padding='same')(max_pool_b1)
max_pool_b2 = MaxPooling2D(pool_size=(2, 2))(conv_b2)
out_b = Flatten()(max_pool_b2)

# the two outputs are merged in fully connected layer
out = concatenate([out_a, out_b])
out = Dropout(0.5)(out)
out = Dense(128, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(10, activation='softmax')(out)

model = Model(inputs, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


filepath = "mnist_cnn.hdf5"

# save weights whenever validation accuracy is improved
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback = [checkpoint]

# fit the model
model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(x_val, y_val), callbacks=callback)