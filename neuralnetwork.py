import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getPred():
    weights = open('cmake-build-release/weights.txt', 'r')
    biases = open('cmake-build-release/biases.txt', 'r')
    architecture = open('cmake-build-release/structure.txt', 'r')

    L = int(architecture.readline())
    sizes = architecture.readline().split(' ')
    for i in range(len(sizes)):
        sizes[i] = int(sizes[i])

    w = [[]]
    b = [[]]

    for line in weights:
        line2 = line.split('^')
        w.append([])
        for i in range(len(line2)):
            line3 = line2[i].split(' ')
            for j in range(len(line3)):
                if (line3[j] == '\n' or line3[j] == ''):
                    line3.pop(j)
                    continue
                line3[j] = float(line3[j])
            w[-1].append(line3)

    for line in biases:
        line2 = line.split(' ')
        for i in range(len(line2)):
            if (line2[i] == '\n' or line2[i] == ''):
                line2.pop(i)
                continue
            line2[i] = float(line2[i])
        b.append(line2)

    inputF = open('number.data', 'r')
    input = inputF.readline().split(' ')
    for i in range(len(input)):
        if (input[i] == '\n'):
            input.pop(i)
            continue
        input[i] = float(input[i])

    # feedforward
    for i in range(1, L):
        newInput = np.zeros(sizes[i])
        for j in range(sizes[i]):
            for k in range(sizes[i - 1]):
                newInput[j] += w[i][j][k] * input[k]
            newInput[j] += b[i][j]
            newInput[j] = sigmoid(newInput[j])

        input = newInput

    # print the index with the highest value
    val = 0
    for i in range(len(input)):
        if (input[i] > input[val]):
            val = i
    print(val)