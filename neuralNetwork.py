# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:35:24 2017

@author: paulo
"""
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

ALPHA = 10.0
my_data = np.genfromtxt('iris.csv', delimiter=';')
#normalizing values
for i in range(5):
    my_data[:,i] = ( my_data[:,i] - min(my_data[:,i]) ) / (max(my_data[:,i]) - min(my_data[:,i]))
#shuffling each part
shuffle(my_data[0:50])
shuffle(my_data[50:100])
shuffle(my_data[100:150])

#separating training set and test set 
trainData = list(my_data[0:35])+ list(my_data[50:85])+ list(my_data[100:135])
testData  = list(my_data[35:50])+ list(my_data[85:100])+ list(my_data[135:150])

#sigmoid function
def sigmoid(x):
   #a = 1.716
   #b = 0.667
   return 1.0 / (1 + np.exp(-x))
   #return ((2.0*a)/(1.0+np.exp(-b*x))) - a

#proccess data
def neuron(entry, weights):
    return np.sum(entry*weights)
#training neural network
def trainNetwork(entries):
    #weights of hidden layer
    weightsH = [None]*3
    for i in range(3):
        weightsH[i] = np.random.uniform(low=-2.4/5,high=2.4/5,size=5)
    #weights of output layer
    weightsOut = np.random.uniform(low=-2.4/4,high=2.4/4,size=4)    
    eqs = []
    while True:    
        errors = []
        #old = [0.0]*4
        #Beta = 0.95
        for entry in entries:       
            xh = np.append(entry[:-1], -1)
            #print xh
            #print weightsH
            yH = [sigmoid(neuron(xh,np.array(weightsH[i]))) for i in range(weightsH.__len__())] 
            xOut = np.append(yH, -1)
            yOut = sigmoid(neuron(xOut, np.array(weightsOut))) 
            
            if yOut!=entry[-1]:
                error = entry[-1] - yOut        
                #gradient and dWeights of out layer
                gradOut = yOut*(1.0-yOut)*error
                dWeightsOut = [ALPHA*xOut[i]*gradOut for i in range(xOut.__len__())]
                
                #with momentum
                #dWeightsOut = [Beta*old[i] + ALPHA*xOut[i]*gradOut for i in range(xOut.__len__())]
                
                #old = dWeightsOut
                #gradient and dWeights of hidden layer
                gradH = [yH[i]*(1.0-yH[i])*gradOut*weightsOut[i] for i in range(yH.__len__())]
                dWeightsH = [None]*weightsH.__len__()
                for i in range(weightsH.__len__()):
                    dWeightsH[i] = [ALPHA*x*gradH[i] for x in xh]
                
                #updating weights
                weightsOut = weightsOut + dWeightsOut
                weightsH = np.matrix(weightsH)+np.matrix(dWeightsH)
                
                errors.append(error)
        eq = sum([x**2 for x in errors])
        eqs.append(eq)
        print eq
        if eq < 0.001:
            break
    return weightsH, weightsOut, eqs

def testNetwork(weightsH, weightsOut, entry):
    xh = np.append(entry[:-1], -1)
    yH = [sigmoid(neuron(xh,np.array(weightsH[i]))) for i in range(weightsH.__len__())] 
    xOut = np.append(yH, -1)
    yOut = sigmoid(neuron(xOut, np.array(weightsOut)))
    return yOut

weightsH, weightsOut, y = trainNetwork(trainData)
print "Weights: "
print weightsH
print weightsOut
x = np.arange(0.0, y.__len__(), 1.0)
plt.plot(x,y)
plt.show()

correct = 0
for entry in testData:
    if (testNetwork(weightsH, weightsOut, entry)-entry[-1]) < 0.05:
        correct += 1
print "Accuracy: "+str(float(correct)*100/testData.__len__())+"%"

