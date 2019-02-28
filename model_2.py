import csv
import cv2
import numpy as np
import pylab
from scipy import ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
lines=[]
with open('./CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines[0]=lines[1]
train_samples, validation_samples = train_test_split(lines, test_size=0.2)        

num=0
correction=0.22
def generator(samples,batch_size=128):
    while 1:
        shuffle(samples)
        for offset in range(0,len(samples),batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images=[]
            left_images=[]
            right_images=[]
            steer_angles=[]
            for line in batch_samples:
       
                image_name=line[0].split('/')[-1]#to get center image name,[1]:left image name,2 right image name
                left_image_name=line[1].split('/')[-1]#left image name
                right_image_name=line[2].split('/')[-1]#right image name
                image=ndimage.imread('./CarND-Behavioral-Cloning-P3/data/IMG/'+image_name)
                left_image=ndimage.imread('./CarND-Behavioral-Cloning-P3/data/IMG/'+left_image_name)
                right_image=ndimage.imread('./CarND-Behavioral-Cloning-P3/data/IMG/'+right_image_name)
                images.append(image)
                left_images.append(left_image)
                right_images.append(right_image)
        
                angle=line[3]
                steer_angles.append(angle)
            steer_angles=[float(i) for i in steer_angles]
            X_train=np.array(images+left_images+right_images)
            left_steer_angles=[float(i)+correction for i in steer_angles]
            right_steer_angles=[float(i)-correction for i in steer_angles]
            y_train=np.array(steer_angles+left_steer_angles+right_steer_angles)
    
            X_train,y_train=preprocess(X_train,y_train)
            #num=len(X_train)
            yield (X_train,y_train)
        
        #outputx=train
    #yield (X_train,y_train)
'''
for line in lines:
    image_name=line[0].split('/')[-1]#to get center image name,[1]:left image name,2 right image name
    left_image_name=line[1].split('/')[-1]#left image name
    right_image_name=line[2].split('/')[-1]#right image name
    image=ndimage.imread('./CarND-Behavioral-Cloning-P3/data/IMG/'+image_name)
    left_image=ndimage.imread('./CarND-Behavioral-Cloning-P3/data/IMG/'+left_image_name)
    right_image=ndimage.imread('./CarND-Behavioral-Cloning-P3/data/IMG/'+right_image_name)
    images.append(image)
    left_images.append(left_image)
    right_images.append(right_image)
    angle=line[3]
    steer_angles.append(angle)
steer_angles=[float(i) for i in steer_angles]
###mix center,left and right images together to get X_train and y_train to preprocess data
X_train=np.array(images+left_images+right_images)
left_steer_angles=[float(i)+correction for i in steer_angles]
right_steer_angles=[float(i)-correction for i in steer_angles]
y_train=np.array(steer_angles+left_steer_angles+right_steer_angles)
'''
#flip images to double input images
def preprocess(X_train,y_train):
    image_flipped=np.fliplr(X_train)
    #print(image_flipped.shape)
    #print(type(y_train[1]))
    y_flipped=-1*y_train
    pre_X_train=np.concatenate([X_train,image_flipped])
    pre_y_train=np.concatenate([y_train,y_flipped])
    #print(pre_X_train.shape)
    #print(pre_y_train.shape)
    return pre_X_train,pre_y_train

#pre_X_train,pre_y_train=preprocess(X_train,y_train)# now data is ready
#exm_img=X_train[20]
#exm_y=np.array([1])
#pre_exm_img=preprocess(exm_img,exm_y)
#plt.imshow(pre_exm_img[0])
#pylab.show()
######
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,MaxPooling2D,Conv2D,Dropout
from keras.layers.convolutional import Convolution2D

model=Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=24, kernel_size=5, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=36, kernel_size=5, padding='valid', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))

model.add(Flatten())
model.add(Dense(240,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(24,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.summary()
#history_object=model.fit(pre_X_train,pre_y_train,validation_split=0.2,shuffle=True,nb_epoch=30,verbose = 1)
train_generator = generator (train_samples)
validation_generator = generator (validation_samples)

history_object=model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/50,epochs=5,validation_data = validation_generator,validation_steps=len(validation_samples)/50,verbose = 1)
model.save('model_2.h5')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
