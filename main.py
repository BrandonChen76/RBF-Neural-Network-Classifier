import pandas as pd
import numpy as np

#test data set
data = pd.DataFrame({
    'Shape': [0, 0, 0.5, 0, 1, 0, 0, 1, 0.5, 0, 1, 0.5],
    'Crust_Size': [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    'Crust_Shade': [0.5, 0, 1, 0, 1, 0, 0.5, 0, 0.5, 1, 0, 0],
    'Filling_Size': [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    'Filling_Shade': [1, 1, 0.5, 1, 0, 1, 0, 0.5, 1, 0, 1, 0.5],
    'Class': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
})

#random weights for perceptron [12 x 2]
weights = np.random.uniform(-0.1, 0.1, size=(12, 2))

#learning rate
n = 0.1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Input: np array of the input [1,5], np array of the c in neuron [1,5]
#Output: the number output
#The activation function for RBF and assuming normalized
def RBF_output (x, c):
    return np.exp(-(np.square(np.linalg.norm(x - c)) / 2))

#Input: np array of the input [1,5]
#Output: np array of the hidden layer [1,12]
#Given the current x input, numpy array; calculate the output array of the nidden layer
def RBF_each (x):

    #each hidden neron has output
    output = np.zeros((1, 12))

    for i, row in data.iterrows():

        #each hidden neuron corresponds to a row input
        c = row[:5].to_numpy()

        #organize the outputs in output
        output[0, i] = RBF_output(x, c)

    return output

#Inputs: the hidden output, np array [1 x 12]; the weights, np array [12 x 2]
#Ouput: prediction y1 and y2 [1 x 2]
def perceptron_prediction (a, b):

    output = np.dot(a, b)

    if np.dot(a, b)[0,0] >= 0:
        output[0,0] = 1
    else:
        output[0,0] = 0
    
    if np.dot(a, b)[0,1] >= 0:
        output[0,1] = 1
    else:
        output[0,1] = 0

    return output

#Inputs: weights; input of hidden layer; prediction; expected
#Ouput: new weights
def update_weights (w, i, p, e):
    ex = np.array([1, 0])
    if e == 0:
        ex = np.array([0, 1])
    return w + (n * np.outer(i.reshape(-1, 1), (ex - p)))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i in range(0, 100, 1):

    correct = 0
    total = 0

    #each epoch
    for i, row in data.iterrows():

        actual_prediction = 0

        #get the data
        input = row.iloc[:5].to_numpy().reshape(1, -1)
        expected = row.iloc[5]

        #do the hidden layer
        hidden_output = RBF_each(input)

        #perceptron
        prediction = perceptron_prediction(hidden_output, weights)

        #weight updates
        weights = update_weights(weights, hidden_output, prediction, expected)

        total += 1

        #get actual prediction
        if prediction[0,0] == 1 and prediction[0,1] == 0:
            actual_prediction = 1
        elif prediction[0,0] == 0 and prediction[0,1] == 1:
            actual_prediction = 0

        #record correct predictions
        if actual_prediction == expected:
            correct += 1

    accuracy = correct / total

#make prediction
X = np.array([[1, 1, 0.5, 0, 0]])
hidden_output = RBF_each(X)
prediction = perceptron_prediction(hidden_output, weights)

if prediction[0,0] == 1 and prediction[0,1] == 0:
    actual_prediction = 1
elif prediction[0,0] == 0 and prediction[0,1] == 1:
    actual_prediction = 0

print(actual_prediction)