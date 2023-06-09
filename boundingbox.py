from PIL import Image

def boundingbox():
    # Open the image file
    image = Image.open("number.jpg")

    # Get the size of the image
    width, height = image.size

    # Access the pixel values
    pixels = list(image.getdata())

    # if every pixel is black, then the image is empty
    if pixels.count((0, 0, 0)) == len(pixels):
        print("empty image")
        return

    for i in range(0, len(pixels)):
        if pixels[i] == (0, 0, 0):
            pixels[i] = 0
        else:
            pixels[i] = 1

    # find the bounding box
    ymin = height
    xmin = width
    ymax = 0
    xmax = 0

    for i in range(0, len(pixels)):
        if pixels[i] == 0:
            continue
        x = i % width
        y = i // width
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)

    if (xmax - xmin) > (ymax - ymin):
        newW = 20
        newH = int((ymax-ymin)/(xmax-xmin)*20)
    else:
        newH = 20
        newW = int((xmax-xmin)/(ymax-ymin)*20)

    # crop the image
    image = image.crop((xmin, ymin, xmax, ymax))

    # resize the image so that the longest side is 20 pixels
    image.thumbnail((newW, newH))

    pixels = list(image.getdata())
    for i in range(0, len(pixels)):
        pixels[i] = pixels[i][0]

    #image.show()

    addX = (28-newW)//2
    addY = (28-newH)//2

    # add black border so that the image is 28x28 and the current is centered
    newPixels = [0 for i in range(28*28)]

    for i in range(0, len(pixels)):
        x = i % newW
        y = i // newW
        newPixels[(y+addY)*28 + x+addX] = pixels[i]

    # save the image to number.data
    file = open("number.data", "w")

    for i in newPixels:
        if int(i)/255 < 0.005:
            file.write("0 ")
        elif int(i)/255 > 0.995:
            file.write("1 ")
        else:
            file.write("{:4.2f} ".format(int(i)/255))
    file.write("\n")
    file.close()