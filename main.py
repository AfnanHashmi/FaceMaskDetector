import cv2
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from keras.models import Sequential
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def MultiClassClassificationModel():
    classifier = Sequential()
    classifier.add(Conv2D(64,kernel_size=(3,3),activation='relu',strides=(1,1)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(100,activation='leaky_relu'))
    classifier.add(Dense(50,activation='relu'))
    classifier.add(Dense(2,activation='softmax'))
    classifier.compile(loss = 'sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'],optimizer = 'adam')
    return classifier

pwfiles = r"Dataset/with_mask/*"
nwfiles = r"Dataset/without_mask/*"

dataset = []
labels = []

for files in glob.glob(pwfiles):
    img = cv2.imread(files)
    dataset.append(img)
    labels.append(0)

for files in glob.glob(nwfiles):
    img = cv2.imread(files)
    dataset.append(img)
    labels.append(1)



dataset = np.array(dataset)
labels = np.array(labels)

X_train,X_test,Y_train,Y_test = train_test_split(dataset,labels,random_state=42,test_size=0.2)

model = MultiClassClassificationModel()

model.fit(X_train,Y_train,epochs = 20,validation_data = (X_test,Y_test))

model.save("FaceMaskdetector.keras")

