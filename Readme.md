# sokobanPlayer
a 2 layer, 2D convolutional network is implemented to recognise the handwritten numbers.

## Inputs:
* 32X32, 2D binary array to represent the handwritten digits.
* 10 Classes ranging representing 0 to 9 handwritten digits.

## Network Architure:
* Simple 2 dimensional convolutional network with 64 and 32 neurons.
* Rectified Linear unit activation function.
* Binary cross-entropy loss function.
* Adam optimization function.
* Dense softmax output layer with output vector size = 10.


## Procedure to view the output,
* Download the "optdigits-orig.windep" file to the directory where your pattern_recognition.py file is present.

### to run the code 
``` 
python3 pattern_recognition.py 

```
