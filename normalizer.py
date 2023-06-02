# transforms the mnist data to a version where the values are normalized to 0-1

# open files
fileIn = open("../stuff/mnist_train.data", "r")
fileOut = open("mnist_train_normalized.data", "w")

for line in fileIn:
    if line == "\n":
        fileOut.write("\n")
        continue

    numbers = line.split(" ")

    fileOut.write(numbers[0] + " ")
    for num in numbers[1:]:
        if int(num)/255 < 0.005:
            fileOut.write("0 ")
        elif int(num)/255 > 0.995:
            fileOut.write("1 ")
        else:
            fileOut.write("{:4.2f} ".format(int(num)/255))
    fileOut.write("\n")

# close files
fileIn.close()
fileOut.close()
