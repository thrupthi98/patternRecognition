#Imports
import numpy as np
from tensorflow.keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape

#Readng the given data file
data_file = open("optdigits-orig.windep")
txt = data_file.read()

#Encoding the data into imputs and targets
data_list = txt.split('\n')[21:]
inputs = []
labels = []
i=0
while i<len(data_list)-32:
  image_list = []
  for j in range(i, 32+i):
    temp_list = [int(v) for v in data_list[j].strip()]
    image_list.append(temp_list)
  labels.append(int(data_list[j+1].strip()))
  i = j+2
  inputs.append(image_list)
inputs = np.array(inputs)

#70% of the inputs are considered for tarianing and the remaining 30% are considered fro testing 
#Pre-processing of training and testing sets
train_set = np.array(inputs[0: 1258]).reshape(-1,32,32,1)
train_labels = np.array(to_categorical(labels[0: 1258]))
test_set = np.array(inputs[1258:]).reshape(-1,32,32,1)
test_labels = np.array(to_categorical(labels[1258:]))

#Creating a 2 layer convolutional neural network using keras library
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compiling the model with adam optimisation function
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#Training the model for the given dataset
#No.of Epochs = 5
#Batch size = 10
model.fit(train_set,train_labels, epochs=5, batch_size=10)

#Testing the model with the testing data and printing the accuracy
_, accuracy = model.evaluate(test_set,test_labels)
print("Model accuracy: %.2f"% (accuracy*100))