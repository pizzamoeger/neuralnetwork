import neuralnetwork

# go through mnist_test_normalized.data and test the neural network
# print the number of correct guesses

# open files
fileIn = open("mnist_test_normalized.data", "r").readlines()

correct = 0
neuralnetwork.load()
count = 0

#neuralnetwork.getPred([0, 0.25, 0.5, 0.75])

for line in fileIn:
    count += 1
    if (count % 100 == 0):
        print(correct/count)
    if line == "\n":
        continue

    numbers = line.split(" ")

    # get the actual number
    actual = int(numbers[0])

    numbers = numbers[1:-1]
    for i in range(len(numbers)):
        numbers[i] = float(numbers[i])

    # get the prediction
    prediction = neuralnetwork.getPred(numbers)[0]

    # check if the prediction is correct
    if int(prediction) == actual:
        correct += 1

print(correct/count)