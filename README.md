# Convolutional-Neural-Network-from-scratch-for-MNIST-Image-Classification
This project creates a convolutional neural network from scratch with no external ML-specific libraries like PyTorch or Tensorflow to provide an insight into how a convolutional neural network operates on a deep level.  
  
_Refer to [Neural-Network-from-scratch-for-MNIST](https://github.com/raisaat/Neural-Network-from-scratch-for-MNIST) in my GitHub to first see how to create a traditional neural network from scratch._


## Summary of the code:
_cnn.py_ is similar to _nn.py_ in [Neural-Network-from-scratch-for-MNIST](https://github.com/raisaat/Neural-Network-from-scratch-for-MNIST) with changes made in the layers constitute the network and the layers' dimensions. Four more classes (the convolution layer class, the max pooling layer class, a class for a layer that performs addition on 3D inputs, and a class for vectorization) were added each with its own forward and back prop functions. The CNN class arranges the layers and does a forward and back propagation on the inputs in a similar way to the NN class.

**Forward path code:** A call to the forward function of the CNN class first normalizes the input. It then passes the input to a for loop that goes through each layer in the network in order and calls each individual layer's forward function, starting with a Convolution layer. The Convolution layer creates a tensor of 3x3 fliters with dimensions 3 x 3 x in x out where in is the input channel dimension and out is the output channel dimension. The filters are initially assigned random values. The layer's forward function then zero-pads the input on each side of its height and width dimensions (the padding on each side is 1 pixel thick). It then performs convolution on the input with the filters with a stride of 1. The max pooling layer's forward function zero pads the input on one side of each of the height and width dimensions (the padding on each side is 1 pixel thick) and then performs max pooling on it. The 3D addition layer's forward function adds biases (intially assigned to 0) to a 3D input. The vectorization layer's forward function flattens the input. The MatrixMulAndAdd layer's forward function multiplies the input vector with the weights (initially assigned random values)and adds biases (intially assigned to 0) to the result. The Relu layer's forward function takes an input and applies the Relu function to it pointwise. The softmax layer's forward function takes an input vector and  applies the softmax function to it pointwise. Each layer's forward function stores the input in a cache and returns the result of the forward pass. After forward pass through each layer, the cnn calculates the loss between the actual output and the network's final output and returns that loss.

**Error code:** Cross entropy was used to calculate the error between the true labels and the predicted labels

**Backward path code:** A call to the backward function of the NN class first calculates the derivative of the cross entropy (the loss) with repect to the network's output and passes that to a for loop that goes through each layer in the network in reverse and calls each individual layer's backward function, starting with the softmax layer. Each layer's backward function takes the derivative of the loss with respect to the layer's ouput (dLoss/dOut) and the learning rate as parameters and calculates the derivative of the loss w.r.t. the previously stored input to the layer (dLoss/dIn) using dLoss/dOut and the derivative of the output w.r.t. the previously stored input to the layer (dOut/dIn); dLoss/dIn = dLoss/dOut * dOut/dIn. The layer finally returns dLoss/dIn, which gets passed to the next layer in line as dLoss/dOut. 

**Weight update code:** If a layer (the convolution layer, the 3D addition layer and the matrix-multiplication-addition layer) has learnable parameters, the layer's backward function also calculates the derivative of the loss w.r.t. each of those parameters (dLoss/dParam) using dLoss/dOut and the derivative of the output w.r.t. the parameters (dOut/dParam); dLoss/dParam = dLoss/dOut * dOut/dParam. It uses stochastic gradient descent to update the parameters using dLoss/dParam and the learning rate that is passed to it: 
Param = Param - lr * dLoss/dParam.

## Architecture:

 **Division by 255.0:** input size = 28 x 28 x 1, output size = 28 x 28 x 1  
 **3x3/1 0 pad Convolution:** input size = 28 x 28 x 1, output size = 28 x 28 x 16, parameter size = 3 x 3 x 1 x 16, MACs = 112896  
 **Addition:** input size = 28 x 28 x 16, output size = 28 x 28 x 16, parameter size = 28 x 28 x 16  
 **ReLU:** input size = 28 x 28 x 16, output size = 28 x 28 x 16  
 **3x3/2 0 pad Max Pool:** input size = 28 x 28 x 16, output size = 14 x 14 x 16  
 **3x3/1 0 pad Convolution:** input size = 14 x 14 x 16, output size = 14 x 14 x 32, parameter size = 3 x 3 x 16 x 32, MACs = 903168  
 **Addition:** input size = 14 x 14 x 32, output size = 14 x 14 x 32, parameter size = 14 x 14 x 32  
 **ReLU:** input size = 14 x 14 x 32, output size = 14 x 14 x 32  
 **3x3/2 0 pad Max Pool:** input size = 14 x 14 x 32, output size = 7 x 7 x 32  
 **3x3/1 0 pad Convolution:** input size = 7 x 7 x 32, output size = 7 x 7 x 64, parameter size = 3 x 3 x 32 x 64, MACs = 903168  
 **Addition:** input size = 7 x 7 x 64, output size = 7 x 7 x 64, parameter size = 7 x 7 x 64  
 **ReLU:** input size = 7 x 7 x 64, output size = 7 x 7 x 64  
 **Vectorization:** input size = 7 x 7 x 64, output size = 3136 x 1  
 **Matrix Multiplication:** input size = 3136 x 1, output size = 100 x 1, parameter size = 3136 x 100, MACs = 313600  
 **Addition:** input size = 100 x 1, output size = 100 x 1, parameter size = 100 x 1  
 **ReLU:** input size = 100 x 1, output size = 100 x 1  
 **Matrix Multiplication:** input size = 100 x 1, output size = 10 x 1, parameter size = 100 x 10, MACs = 1000  
 **Addition:** input size = 10 x 1, output size = 10 x 1, parameter size = 10 x 1  
 **Softmax:** input size = 10 x 1, output size = 10 x 1
 
 ## Performance Display:
 `Total execution time: 3782.25273723173909 minutes`

## Instructions on running the code:

1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
2. File - New Python 3 notebook
3. Cut and paste this file into the cell (feel free to divide into multiple cells)
4. Runtime - Run all
