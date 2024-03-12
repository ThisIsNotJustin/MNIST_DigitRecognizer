import os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import gradio as gr

# read and process data
data = pd.read_csv(os.path.join(os.getcwd(), 'mnist', 'train.csv'))
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# training data
train = data[1000:m].T
yTrain = train[0]
xTrain = train[1:n].astype(float)
xTrain /= 255
# testing data
test = data[0:1000].T
yTest = test[0]
xTest = test[1:n].astype(float)
xTest /= 255



# weights and biases of layers 1 and 2
def init_params():
    W1 = np.random.randn(10, 784) * 0.05
    b1 = np.random.randn(10, 1) * 0.05
    W2 = np.random.randn(10, 10) * 0.05
    b2 = np.random.randn(10, 1) * 0.05
    return W1, b1, W2, b2

# ReLU Activation Function (Hidden Layer)
# any positive value in Z is itself and any negative value is 0
def relu(Z):
    return np.maximum(Z, 0)

# derivative of ReLU Function
# if z > 0, returns true (1)
# if z < 0, returns false (0)
def deriv_relu(Z):
    return Z > 0

# Softmax Activation Function
def softmax(Z):
    Z -= np.max(Z, axis=0) # subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0) # z^2 / sum of all z^2
    return A

# returns a matrix of one-hot encoded vectors
def one_hot_encode(Y):
    hotY = np.zeros((Y.size, Y.max() + 1)) # creates 10x10 matrix of zeros from vector Y
    hotY[np.arange(Y.size), Y] = 1 
    hotY = hotY.T
    return hotY

# forward propagation through neural network given defined weights and biases
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1) # relu on Z1
    Z2 = W2.dot(A1) + b2 
    A2 = softmax(Z2) # softmax on Z2
    return Z1, A1, Z2, A2

# backward propagation to compute gradients of the loss to respective variables and parameters
# returns gradients of loss to biases and weights of the hidden and output layer
def backward_prop(Z1, A1, A2, W2, X, Y):
    hotY = one_hot_encode(Y)
    dZ2 = A2 - hotY
    e = 1e-8
    db2 = 1 / m * np.sum(dZ2, axis=1).reshape(-1,1)
    dW2 = 1 / m * dZ2.dot(A1.T)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    db1 = 1 / m * np.sum(dZ1, axis=1).reshape(-1,1)
    dW1 = 1 / m * dZ1.dot(X.T)
    return dW1, db1, dW2, db2

# updates the current weights and biases using the learning rate, alpha, and the updated weights
# and biases from backward_prop
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10, 1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10, 1))
    return W1, b1, W2, b2

# predicts digit based on highest probability
def predict(A2):
    return np.argmax(A2, 0)

# divides num of correct predictions with total samples
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# train model for making predictions
# prints iteration and accuracy every 50 iterations
def gradient_desc(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            prediction = predict(A2)
            print(get_accuracy(prediction, Y))
    return W1, b1, W2, b2

# train model
W1, b1, W2, b2 = gradient_desc(xTrain, yTrain, .1, 500)

def getWb():
    return W1, b1, W2, b2

def make_prediction(X, W1, b1, W2, b2):
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
        prediction = predict(A2)
        return prediction
    
# graphically show digit, prediction, and label
def test_prediction(index, W1, b1, W2, b2):
    curr = xTest[:, index, None]
    prediction = make_prediction(curr, W1, b1, W2, b2)
    label = yTest[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    curr = curr.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(curr, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)

# test neural network
test = make_prediction(xTest, W1, b1, W2, b2)
print(get_accuracy(test, yTest))


# testing custom input given a black/white canvas from gradio
def canvas_prediction(canvas_image):
    W1, b1, W2, b2 = getWb()
    image = canvas_image.reshape((28, 28)).astype('float32') / 255
    image = image.reshape(-1,1)
    prediction = make_prediction(image, W1, b1, W2, b2)
    return int(prediction[0])


demo = gr.Interface(
    fn=canvas_prediction,
    inputs=gr.Image(shape=(28, 28), invert_colors=True, image_mode="L", source="canvas"),
    outputs=gr.Label(),
)

demo.launch(share='True')