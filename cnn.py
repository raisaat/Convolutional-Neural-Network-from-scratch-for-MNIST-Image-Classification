
################################################################################
#
# LOGISTICS
#
#    Name: Raisaat Rashid
#
# FILE
#
#    cnn.py
#
# DESCRIPTION
#
#    MNIST image classification with a CNN written and trained in Python
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

import os.path
import urllib.request
import gzip
import math
import numpy             as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

DATA_BATCH_SIZE  = 25

# training (linear warm up with cosine decay learning rate)
TRAINING_LR_MAX          = 0.001
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 3
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 6
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

class Conv3:
  # Convolution layer with 3x3 filters and stride 2.

  def __init__(self, in_channels, out_channels):
    self.filters_in_channels = in_channels
    self.filters_out_channels = out_channels
    self.filters = np.random.randn(3, 3, in_channels, out_channels) / 9.0

  def get_region(self, input):
    '''
    Generates 3x3 regions in the image for the filters.
    '''
    h, w, _ = input.shape

    for i in range(h - 2):
      for j in range(w - 2):
        region = input[i:(i + 3), j:(j + 3), :] # Generate the region
        yield region, i, j

  def forward(self, input):
    '''
    Executes a forward pass of the conv3 layer on the input.
    '''
    h, w, _ = input.shape
    output = np.zeros((h, w, self.filters_out_channels))

    input = np.pad(input, ((1, 1), (1, 1), (0, 0)), 'constant') # zero-pad the image on each side of the height and width dimensions
    self.cache = input # store input for back propagation

    for f in range(self.filters_out_channels):
      for region, i, j in self.get_region(input):
        output[i, j, f] = np.sum(region * self.filters[:, :, :, f]) # perform convolution

    return output
  
  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the conv3 layer.
    '''
    dLoss_df = np.zeros(self.filters.shape)
    dLoss_dInput = np.zeros(self.cache.shape)

    for region, i, j in self.get_region(self.cache):
      h, w, ch = region.shape
      for f in range(self.filters_out_channels):
        dLoss_df[:, :, :, f] = dLoss_df[:, :, :, f] + dLoss_dOut[i, j, f] * region # Compute dLoss/dW where W is the weight matrix of the filters
        for i2 in range(h):
          for j2 in range(w):
            for c in range(ch):
              dLoss_dInput[i + i2, j + j2, c] = dLoss_dInput[i + i2, j + j2, c] + dLoss_dOut[i, j, f] * self.filters[i2, j2, c, f] # Compute dLoss/dInput where Input is the input to this layer

    # Update filters
    self.filters = self.filters - lr * dLoss_df

    # drop the padding in the width dimension
    dLoss_dInput = np.delete(dLoss_dInput, -1, axis=1)
    dLoss_dInput = np.delete(dLoss_dInput, 0, axis=1)

     # drop the padding in the height dimension
    dLoss_dInput = np.delete(dLoss_dInput, -1, axis=0)
    dLoss_dInput = np.delete(dLoss_dInput, 0, axis=0)

    return dLoss_dInput

class MaxPool3:
  # Max Pooling layer with pool size of 3 and stride 2.

  def get_region(self, input):
    '''
    Generates 3x3 regions in the image for the filters.
    '''
    h, w, _ = input.shape
    h2 = h // 2
    w2 = w // 2

    for i in range(h2):
      for j in range(w2):
        region = input[(i * 2):(i * 2 + 3), (j * 2):(j * 2 + 3)] # Generate the region
        yield region, i, j

  def forward(self, input):
    '''
    Executes a forward pass of the Max Pool layer on the input.
    '''
    h, w, ch = input.shape
    output = np.zeros((h // 2, w // 2, ch))

    input = np.pad(input, ((0, 1), (0, 1), (0, 0)), 'constant') # zero-pad the image on one side in each of the width and height dimensions
    self.cache = input

    for region, i, j in self.get_region(input):
      output[i, j] = np.amax(region, axis=(0, 1)) # get the max value of the region

    return output
  
  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the max pool layer.
    '''
    dLoss_dInput = np.zeros(self.cache.shape)

    for region, i, j in self.get_region(self.cache):
      h, w, f = region.shape
      amax = np.amax(region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            if region[i2, j2, f2] == amax[f2]:
              dLoss_dInput[i * 2 + i2, j * 2 + j2, f2] = dLoss_dOut[i, j, f2] # send the gradient to the max element in the region
    
    dLoss_dInput = np.delete(dLoss_dInput, -1, axis=1) # drop the padding in the width dimension
    dLoss_dInput = np.delete(dLoss_dInput, -1, axis=0) # drop the padding in the height dimension
    return dLoss_dInput

class Addition3D:
  # Addition layer for 3D inputs
  def __init__(self, h, w, num_filters):
    self.bias = np.zeros((h, w, num_filters))
  
  def forward(self, input):
    '''
    Executes a forward pass of the addition layer on the input.
    '''
    return input + self.bias
  
  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the addition layer.
    '''
    self.bias = self.bias - lr * dLoss_dOut
    return dLoss_dOut

class Relu:
  # Relu layer to perform relu operation on inputs
  def forward(self, input):
    '''
    Executes a forward pass of the relu layer on the input.
    '''
    self.cache = input
    return np.maximum(0, input)

  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the relu layer.
    '''
    dOut_dIn = self.cache
    dOut_dIn[dOut_dIn <= 0] = 0
    dOut_dIn[dOut_dIn > 0] = 1

    return dLoss_dOut * dOut_dIn # return the dLoss/dIn where In is the input to the layer

class Vectorization:
  # Vectorization layer to perform vectorization
  def forward(self, input):
    '''
    Executes a forward pass of the vectorization layer on the input.
    '''
    self.cache = input.shape
    return input.flatten()

  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the vectorization pool layer.
    '''
    return dLoss_dOut.reshape(self.cache)

class MatrixMulAndAdd:
  # Layer to perform matrix multiplication and addition on 2D inputs
  def __init__(self, input_dim, out_dim):
    self.weights = np.random.randn(input_dim, out_dim) / input_dim
    self.bias = np.zeros(out_dim)
  
  def forward(self, input):
    '''
    Executes a forward pass of the matrix multiplication and addition layer on the input.
    '''
    self.cache = input
    return np.dot(input, self.weights) + self.bias
  
  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the matrix multiplication and addition layer.
    '''
    dOut_dW = self.cache
    dOut_dIn = self.weights

    # Calculate the derivatives
    dLoss_dW = dOut_dW[np.newaxis].T @ dLoss_dOut[np.newaxis]
    dLoss_dIn = dOut_dIn @ dLoss_dOut

    # update weights and bias
    self.weights = self.weights - lr * dLoss_dW
    self.bias = self.bias - lr * dLoss_dOut

    return dLoss_dIn

class Softmax:
  # Softmax layer to perform softmax on the input  
  def forward(self, input):
    '''
    Executes a forward pass of the softmax layer on the input.
    '''
    self.cache = input
    exp = np.exp(input)
    return exp / np.sum(exp, axis=0)

  def backward(self, dLoss_dOut, lr):
    '''
    Performs a backward pass of the softmax layer.
    '''
    for i, derivative in enumerate(dLoss_dOut):
      if derivative == 0:
        continue

      t_exp = np.exp(self.cache)
      S = np.sum(t_exp)

      # Compute the derivatives
      dOut_dIn = -t_exp[i] * t_exp / (S ** 2)
      dOut_dIn[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
      return derivative * dOut_dIn

class ConvNeuralNet:
  # A Convolutional Neural Network
  def __init__(self):
    self.layers = [
        Conv3(1, 16),
        Addition3D(28, 28, 16),
        Relu(),
        MaxPool3(),
        Conv3(16, 32),
        Addition3D(14, 14, 32),
        Relu(),
        MaxPool3(),
        Conv3(32, 64),
        Addition3D(7, 7, 64),
        Relu(),
        Vectorization(),
        MatrixMulAndAdd(3136, 100),
        Relu(),
        MatrixMulAndAdd(100, 10),
        Softmax()
    ]
    self.cache = {}
  
  def forward(self, X, label):
    '''
    Executes a forward pass of all the layers on the input.
    '''
    X = np.divide(X, 255.0) # normalize the input
    self.true_label = label

    for layer in self.layers: # do a forward pass
      X = layer.forward(X)
    
    self.Yh = X
    loss = -np.log(self.Yh[label]) # compute the loss

    return loss
  
  def backward(self, lr):
    '''
    Executes a backward pass of all the layers.
    '''

    # Calculate dLoss/dYh
    derivative = np.zeros(10)
    derivative[self.true_label] = -1 / self.Yh[self.true_label]

    for layer in reversed(self.layers): # do a backward pass
      derivative = layer.backward(derivative, lr)

  def predict(self, X, label):
    '''
    Makes a prediction of the output label given an input
    '''
    loss = self.forward(X, label)
    prediction = np.argmax(self.Yh)
    return prediction

# learning rate schedule
def lr_schedule(epoch):
  # linear warmup followed by cosine decay
  if epoch < TRAINING_LR_INIT_EPOCHS:
    lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
  else:
    lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL
  return lr

# start epoch
start_epoch = 0

cnn = ConvNeuralNet() # Define a CNN
epochs = []
accuracies = []
for epoch in range(start_epoch, TRAINING_NUM_EPOCHS):
  epochs.append(epoch)

  training_loss = 0.0
  accuracy_sum = 0.0
  lr = lr_schedule(epoch)

  # train
  for i in range(DATA_NUM_TRAIN):
    loss = cnn.forward(train_data[i].reshape(DATA_ROWS, DATA_COLS, 1), train_labels[i])
    cnn.backward(lr)
    training_loss = training_loss + loss
  
  # test
  test_correct = 0.0
  for j in range(DATA_NUM_TEST):
     test_correct = test_correct + (cnn.predict(test_data[j].reshape(DATA_ROWS, DATA_COLS, 1), test_labels[j]) == test_labels[j])
  
  accuracy = 100.0*(test_correct/DATA_NUM_TEST)
  accuracies.append(accuracy)
  # epoch statistics
  print('Epoch {0:2d} lr = {1:8.6f} avg loss = {2:8.6f} accuracy = {3:5.2f}'.format(epoch, lr, training_loss/DATA_NUM_TRAIN, accuracy))

################################################################################
#
# DISPLAY
#
################################################################################

# performance display

# test with the final model
test_correct = 0.0
predictions = []
for j in range(DATA_NUM_TEST):
  prediction = cnn.predict(test_data[j].reshape(DATA_ROWS, DATA_COLS, 1), test_labels[j])
  test_correct = test_correct + (prediction == test_labels[j])
  predictions.append(prediction)

# test set statistics
print('Final accuracy of test set = {0:5.2f}'.format((100.0*test_correct/DATA_NUM_TEST)))
plt.plot(epochs, accuracies)
plt.xlabel("epoch")
plt.ylabel("Accuracy (%)")

# example display
# replace the xNN predicted label with the label predicted by the network
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(predictions[i]))
    plt.imshow(img, cmap='Greys')
plt.show()

print("\nTotal execution time: %s minutes" % ((time.time() - start_time)/60))
