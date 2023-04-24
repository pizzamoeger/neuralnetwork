# transforms the mnist data to a version where the values are normalized to 0-1

# open files
fileIn = open("../data/mnist_test.csv", "r")
fileOut = open("mnist_test_normalized.data", "w")

for line in fileIn:
    if line == "\n":
        fileOut.write("\n")
        continue

    numbers = line.split(",")

    fileOut.write(numbers[0] + " ")
    for num in numbers[1:]:
        fileOut.write(str(int(num)/255) + " ")
    fileOut.write("\n")

# close files
fileIn.close()
fileOut.close()
