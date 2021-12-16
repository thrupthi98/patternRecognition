# Pattern Recognition
A 2 layer, 2D convolutional network is implemented to recognise the handwritten numbers.

## Inputs:
* 32X32, 2D binary array to represent the handwritten digits.
* 10 Classes ranging representing 0 to 9 handwritten digits.
* Total images 1797.
* Training examples - 70% ~ 1258.

## Network Architure:
* Simple 2 dimensional convolutional network with 64 and 32 neurons.
* Rectified Linear unit activation function.
* Binary cross-entropy loss function.
* Adam optimization function.
* Dense softmax output layer with output vector size = 10.

## Procedure to view the output:
* Download the "optdigits-orig.windep" file to the directory where your pattern_recognition.py file is present.

## Output:
* First, the accuracy and loss of the model is printed after each epoch.
* Second, the test accuracy in percentage is represented for the trained model.

## Experiments:
* Changes to epoch and batch size can be performed by altering the respective values on line 50.
```
model.fit(train_set,train_labels, epochs=5, batch_size=10)
```
* Number of neuraons can be changed by altering the first parameters of Conv2D function on line 39 and 40.
```
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
```
* The number of training examples and testing examples can be performed by changing the number 1258 on lines 32 to 35.
```
train_set = np.array(inputs[0: 1258]).reshape(-1,32,32,1)
train_labels = np.array(to_categorical(labels[0: 1258]))
test_set = np.array(inputs[1258:]).reshape(-1,32,32,1)
test_labels = np.array(to_categorical(labels[1258:]))
```

### to run the code 
``` 
python3 pattern_recognition.py 
```
