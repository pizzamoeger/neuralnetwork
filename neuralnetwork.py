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
    inputN = inputF.readline().split(' ')
    for i in range(len(inputN)):
        if (inputN[i] == '\n'):
            inputN.pop(i)
            continue
        inputN[i] = float(inputN[i])
    inputO = inputN

    # feedforward
    for i in range(1, L):
        newInput = np.zeros(sizes[i])
        for j in range(sizes[i]):
            for k in range(sizes[i - 1]):
                newInput[j] += w[i][j][k] * inputN[k]
            newInput[j] += b[i][j]
            newInput[j] = sigmoid(newInput[j])

        inputN = newInput

    # print the index with the highest value
    val = 0
    for i in range(len(inputN)):
        if (inputN[i] > inputN[val]):
            val = i

    print(val)
    stp = input("add to training data? (0/1)")
    if (stp == '1'):
        label = input("correct label: ")


        f = open('cmake-build-release/mnist_train_normalized.data', 'a')
        f.write(label + ' ')
        for i in range(len(inputO)):
            f.write(str(inputO[i]) + ' ')
        f.write('\n')
        f.close()

    weights.close()
    biases.close()
    architecture.close()
    inputF.close()