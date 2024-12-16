import numpy as np
import pickle
from random import random


def sigmoid(num):
    return(1/(1+np.exp(-num)))
sigmoid_vectorized = np.vectorize(sigmoid)


def activationFuncDerivative(activationFunc,x):
    return activationFunc(x)*(1-activationFunc(x))

def test(inputs, w, b, activationFunc):
    correct = 0
    for inp in inputs: #inputs (0,1 stuff like that)
        As ={}
        As[0] = inp[0]
        dots = {}        
        for layer in range(1, len(w)): #get layer
            dots[layer] = (w[layer]@As[layer-1])+b[layer]
            As[layer] = activationFunc(dots[layer])
        maxI = 0
        for i in range(10):
            if(As[len(w)-1][i, 0] > As[len(w)-1][maxI, 0]):
                maxI = i
        if(inp[1][maxI, 0] == 1):
            correct +=1 
    return correct
       
def back_propagation(inputs, w, b, activationFunc, learningRate, epochs):
    for epoch in range(epochs):     
        for ind, inp in enumerate(inputs): #inputs (0,1 stuff like that)
            if(ind%2000 == 0):
                print(ind)
            As ={}
            As[0] = inp[0]
            dots = {}        
            for layer in range(1, len(w)): #get layer
                dots[layer] = (w[layer]@As[layer-1])+b[layer]
                As[layer] = activationFunc(dots[layer])
            deltas= {}
            deltas[len(w)-1] = activationFuncDerivative(activationFunc, dots[len(w)-1]) * (inp[1]-As[len(w)-1])
            
            for layer in range(len(w)-2, 0,-1):
                deltas[layer] = activationFuncDerivative(activationFunc, dots[layer]) *(np.transpose(w[layer+1])@deltas[layer+1])
            for layer in range(1, len(w)):
                b[layer] = b[layer]+learningRate*deltas[layer]
                w[layer] = w[layer]+learningRate*deltas[layer] *np.transpose(As[layer-1])
            
    return (w,b)

inputs = []
with open("mnist_train.csv") as f:
    for line in f:
        splitLine = line.split()
        commaSplitLine = splitLine[0].split(",")
        target = int(commaSplitLine[0])
        solution = np.zeros((10, 1))
        solution[target, 0] = 1
        if(random() <0.3): #distortions
            inputValues = np.zeros((784, 1))
            for i in range(2, len(commaSplitLine)+1):
                inputValues[i-2, 0] = int(commaSplitLine[i-1])/255
        else:
            inputValues = np.zeros((784, 1))
            for i in range(1, len(commaSplitLine)):
                inputValues[i-1, 0] = int(commaSplitLine[i])/255
        inputs.append((inputValues, solution))
# print(inputs)
tests = []
with open("mnist_test.csv") as f:
    for line in f:
        splitLine = line.split()
        commaSplitLine = splitLine[0].split(",")
        target = int(commaSplitLine[0])
        solution = np.zeros((10, 1))
        solution[target, 0] = 1
        inputValues = np.zeros((784, 1))
        for i in range(1, len(commaSplitLine)):
            inputValues[i-1, 0] = int(commaSplitLine[i])/255
        tests.append((inputValues, solution))

def create_rand_values(dimensions):
    weights= [None]
    biases = [None]
    for i in range(1,len(dimensions)):
        weights.append(2*np.random.rand(dimensions[i],dimensions[i-1]) - 1)
        biases.append(2*np.random.rand(dimensions[i],1)-1)
    return weights, biases



with open("weights_and_biases.pkl", "rb") as f:
    w1,b1 = pickle.load(f)
print(w1)
print(test(tests, w1, b1, sigmoid))

for i in range(1):
    w1, b1 = back_propagation(inputs, w1, b1, sigmoid, 0.1, 1)

    with open("weights_and_biases.pkl", "wb") as f:
        pickle.dump((w1,b1), f)
    print(test(tests, w1, b1, sigmoid))
    print("run number " + str(i))
print("done")

