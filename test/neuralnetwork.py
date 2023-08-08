import math
import numpy as np

layer_data = []
L = 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0.0, x)

def loadFullyConn(data):
    out = [int(data[0]), 1, 1]
    inn = layer_data[-2][-1]
    newinn = 1
    for i in inn:
        newinn *= i
    inn = [newinn, 1, 1]

    data[1] = data[1].split(" ")
    biases = []
    for bias in data[1][:-1]:
        biases.append(float(bias))

    data[2] = data[2].split(" ")
    biasesVelocity = []
    for biasVel in data[2][:-1]:
        biasesVelocity.append(float(biasVel))

    data[3] = data[3].split(" ")
    weights = []
    for weight in data[3][:-1]:
        weights.append(float(weight))

    data[4] = data[4].split(" ")
    weightsVelocity = []
    for weightVel in data[4][:-1]:
        weightsVelocity.append(float(weightVel))

    layer_data.append([biases, weights, inn, out])

def loadConvLayer(data):
    data[0] = data[0].split(" ")
    strideLength = int(data[0][0])
    receptiveFieldLength = int(data[0][1])
    featureMaps = int(data[0][2])
    inn = layer_data[-2][-1]
    out = [int((inn[0]-receptiveFieldLength+1)/strideLength), int((inn[1]-receptiveFieldLength+1)/strideLength), featureMaps]

    data[1] = data[1].split(" ")
    biases = []
    for bias in data[1][:-1]:
        biases.append(float(bias))

    data[2] = data[2].split(" ")
    biasesVelocity = []
    for biasVel in data[2][:-1]:
        biasesVelocity.append(float(biasVel))

    data[3] = data[3].split(" ")
    weights = []
    for weight in data[3][:-1]:
        weights.append(float(weight))

    data[4] = data[4].split(" ")
    weightsVelocity = []
    for weightVel in data[4][:-1]:
        weightsVelocity.append(float(weightVel))

    layer_data.append([biases, weights, strideLength, receptiveFieldLength, inn, out])

def loadMaxPoolLayer(data):
    inn = layer_data[-2][-1]
    out = [int(inn[0]/int(data[0])), int(inn[1]/int(data[0])), inn[2]]
    layer_data.append([int(data[0]), out])

def loadInput(data):
    data[0] = data[0].split(" ")
    layer_data.append([[int(data[0][0]), int(data[0][1]), 1]])

def ffFullConn(l, a):
    biases = layer_data[l][0]
    weights = layer_data[l][1]
    inn = layer_data[l][2]
    out = layer_data[l][3]

    z = [0] * out[0]
    newA = z

    for neuron in range(out[0]):
        for previous_neuron in range(inn[0]):
            z[neuron] += weights[neuron*previous_neuron+previous_neuron]*a[previous_neuron]
        z[neuron] += biases[neuron]
        if (l == len(layer_data)-1):
            newA[neuron] = sigmoid(z[neuron])
        else:
            newA[neuron] = relu(z[neuron])
    return newA

def dataIndex(map, y, x, noutx, nouty):
    return map*noutx*nouty + y*noutx + x

def convweightIndex(prevmap, map, y, x, recf, noutfm):
    return prevmap* (noutfm*recf*recf) + map * (recf*recf) + y*recf + x

def ffConv(l, a):
    biases = layer_data[l][0]
    weights = layer_data[l][1]
    strideLength = layer_data[l][2]
    receptiveFieldLength = layer_data[l][3]
    featureMaps = layer_data[l][-1][2]
    noutx = layer_data[l][-1][0]
    nouty = layer_data[l][-1][1]
    prevnoutx = layer_data[l-2][-1][0]
    prevnouty = layer_data[l-2][-1][1]

    z = [0] * featureMaps*noutx*nouty
    newA = z

    for map in range(featureMaps):
        for y in range(nouty):
            for x in range(noutx):
                for prevmap in range(layer_data[l-2][-1][2]):
                    for kerny in range(receptiveFieldLength):
                        for kernx in range(receptiveFieldLength):
                            z[dataIndex(map,y,x,noutx,nouty)] += weights[convweightIndex(prevmap, map, kerny, kernx, receptiveFieldLength, featureMaps)]*a[dataIndex(prevmap, y*strideLength+kerny, x*strideLength+kernx, prevnoutx, prevnouty)]
                z[dataIndex(map,y,x,noutx,nouty)] += biases[map]
                newA[dataIndex(map,y,x,noutx,nouty)] = relu(z[dataIndex(map,y,x,noutx,nouty)])

    return newA

def ffMaxPool(l, a):
    summarizedRegLen = layer_data[l][0]
    out = layer_data[l][1]
    prevout = layer_data[l-2][-1]

    z = [-1000000] * out[0]*out[1]*out[2]

    for map in range(out[2]):
        for y in range(out[1]):
            for x in range(out[0]):
                for kerny in range(summarizedRegLen):
                    for kernx in range(summarizedRegLen):
                        z[dataIndex(map, y, x, out[0], out[1])] = max(z[dataIndex(map, y, x, out[0], out[1])], a[dataIndex(map, y*summarizedRegLen+kerny, x*summarizedRegLen+kernx, prevout[0], prevout[1])])
    return z
def ff(l, activations):
    if layer_data[l] == 0:
        return ffFullConn(l+1, activations)
    elif layer_data[l] == 1:
        return ffConv(l+1, activations)
    elif layer_data[l] == 2:
        return ffMaxPool(l+1, activations)

def load(filename):
    network = open(filename, 'r').readlines()

    L = int(network[0][0])
    for l in range(L):
        data = network[l+1].split("\n")
        data = data[:-1]
        for i in range(len(data)):
            data[i] = data[i].split("//")
        data = data[0]
        type = int(data[0])
        layer_data.append(type)
        if type == 3:
            loadInput(data[1:])
        elif type == 1:
            loadConvLayer(data[1:])
        elif type == 2:
            loadMaxPoolLayer(data[1:])
        else:
            loadFullyConn(data[1:])

def getPred(inputN):
    inputO = inputN

    # feedforward
    for i in range(1, int(len(layer_data)/2)):
        inputN = ff(2*i, inputN)

    # print the index with the highest value
    val = 0
    for i in range(len(inputN)):
        if (inputN[i] > inputN[val]):
            val = i

    return [val, inputO]