import tensorflow as tf
import cv2, numpy as np, time, glob
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, add, concatenate
from keras.backend.tensorflow_backend import set_session

# limit tensorflow gpu usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



# define the cnn network and load the weights

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

out = concatenate([out_a, out_b])
out = Dropout(0.5)(out)
out = Dense(128, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(10, activation='softmax')(out)

model = Model(inputs, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# load the trained weights
model.load_weights('mnist_cnn.hdf5')




# initialize camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('camera')


# for inbuilt camera and blue cap
# lower = np.array([100,  50, 50])
# upper = np.array([120, 255, 255])
# image = np.uint8(np.ones((480, 640, 3))) * 255


# for inbuilt camera and cyan cap
lower = np.array([80,  50, 50])
upper = np.array([110, 255, 255])
image = np.uint8(np.ones((480, 640, 3))) * 255


# for external camera and blue cap
# lower = np.array([110,  50, 50])
# upper = np.array([130, 255, 255])
# image = np.uint8(np.ones((480, 640, 3)))*255

# for external camera and cyan cap
# lower = np.array([90,  50, 50])
# upper = np.array([120, 255, 255])
# image = np.uint8(np.ones((480, 640, 3)))*255



print ('--------------------------------------------------')
print ('--------------------------------------------------')
print ('--------------------------------------------------')
print ('Press 1 to start drawing.')
print ('Press 2 to finish drawing and generate prediction.')
print ('Press Esc to end program')
print ('--------------------------------------------------')

flag = 0
cnt = 0

while(1):    

    _, frame = cap.read()
    img  = frame.copy()

    # rgb2hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # black and white mask using hsv thresholding
    mask = cv2.inRange(hsv, lower, upper)

    # process image
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # if circle found then
    if len(cnts) > 0:

        # get only maximum circle beyond certain threshold
        c = sorted(cnts, key = cv2.contourArea, reverse=True)
        ((x1, y1), radius1) = cv2.minEnclosingCircle(c[0])
        
        if radius1 > 25:                          

            # draw circle in camera
            cv2.circle(img, (int(x1), int(y1)), int(radius1), (0, 255, 255), 2)


            # draw the image
            if flag == 1:
                cv2.circle(image, (int(x1), int(y1)), 25, 0, thickness=-1, lineType=8)
        
        
    
    cv2.imshow('camera', cv2.flip(img, 1))
    cv2.imshow('mask', cv2.flip(mask, 1))
    cv2.imshow('image', cv2.flip(image, 1))
    k = cv2.waitKey(5) & 0xFF

    # press 1 to start drawing
    if k == 49:
        flag = 1
    

    # press 2 to stop drawing and print prediction
    elif k == 50:
        flag = 0
        cv2.imwrite('captured/' + str(cnt) +'.jpg', cv2.resize(cv2.flip(image, 1), (28, 28)))
        image = np.uint8(np.ones((480, 640, 3))) * 255
        img_file = 'captured/' + str(cnt) + '.jpg'
        

        im = cv2.imread(img_file, 0)
        im = (255 - im)/255.0
        
        # reshape the image and generate the prediction
        im = im.reshape((28, 28, 1))
        
        predictions = model.predict_classes(np.array([im]))[0]
        print ('Detected number is:', predictions)
        cnt = cnt + 1   

    # press esc to close the windows    
    elif k == 27:
        break
         
cv2.destroyAllWindows()