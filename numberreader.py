import pygame
from pygame.locals import *
import numpy as np
import boundingbox
import neuralnetwork

neuralnetwork.load()

pygame.init()
screen = pygame.display.set_mode((700, 700))
pygame.display.set_caption("Number")

# set background to white
background = pygame.Surface(screen.get_size())
background = background.convert()

# blit background to screen
screen.blit(background, (0, 0))
pygame.display.flip()
draw = False

"""
stp = input("add to training data? (0/1)")
label = input("correct label: ")

f = open('cmake-build-release/mnist_train_normalized.data', 'a')
f.write(label + ' ')
for i in range(len(inputO)):
f.write(str(inputO[i]) + ' ')
f.write('\n')
f.close()
"""



while True:
    # all but the upper 350x350 square is white, the upper 350x350 square is black
    screen.fill((255, 255, 255))

    # the top is a grey title bar containing the text "Neural Network to recognize handwritten digits"
    pygame.draw.rect(screen, (200, 200, 200), (0, 0, 700, 50))
    font = pygame.font.SysFont("Arial", 30)
    text = font.render("Neural Network to recognize handwritten digits", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    # the bottom are two grey bars
    # the upper one contatining the text "Press enter to let the network guess the number"
    # and below this "Press escape to exit"
    pygame.draw.rect(screen, (200, 200, 200), (0, 550, 700, 150))

    font = pygame.font.SysFont("Arial", 30)
    text = font.render("Press enter to let the network guess the number", True, (0, 0, 0))
    screen.blit(text, (10, 560))

    font = pygame.font.SysFont("Arial", 30)
    text = font.render("Press c to guess a new number", True, (0, 0, 0))
    screen.blit(text, (10, 610))

    font = pygame.font.SysFont("Arial", 30)
    text = font.render("Press escape to exit", True, (0, 0, 0))
    screen.blit(text, (10, 660))

    # on the middle left is a 350x350 black square
    pygame.draw.rect(screen, (0, 0, 0), (0, 50, 350, 350))

    # on the middle right add the text "Prediction:"
    font = pygame.font.SysFont("Arial", 30)
    text = font.render("Prediction:", True, (0, 0, 0))
    screen.blit(text, (400, 50))

    while True:
        # let the user draw on the screen
        # resolution is 28x28
        # position of the mouse
        m = pygame.mouse.get_pos()
        if (draw):
            # only if the mouse is in the black square
            if (m[0] < 350 and m[1] < 400 and m[0] > 0 and m[1] > 50):
                pygame.draw.circle(screen, (255, 255, 255), m, 15)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
            elif event.type == MOUSEBUTTONDOWN:
                draw = True
            elif event.type == MOUSEBUTTONUP:
                draw = False
            elif event.type == KEYDOWN:
                if event.key == K_RETURN:
                    # save the 350x350 blqck square to number.jpg
                    pygame.image.save(screen.subsurface(0, 50, 350, 350), "number.jpg")

                    boundingbox.boundingbox()

                    inputF = open('number.data', 'r')
                    inputN = inputF.readline().split(' ')
                    for i in range(len(inputN)):
                        if (inputN[i] == '\n'):
                            inputN.pop(i)
                            continue
                        inputN[i] = float(inputN[i])
                    inputF.close()

                    num = neuralnetwork.getPred(inputN)
                    pred = num[0]

                    # print the pred next to the text "Prediction:"
                    font = pygame.font.SysFont("Arial", 30)
                    text = font.render(str(pred), True, (0, 0, 0))
                    screen.blit(text, (520, 50))
                    pygame.display.flip()

                elif event.key == K_ESCAPE:
                    pygame.quit()
                    exit()
                elif event.key == K_c:
                    # clear the black square and the prediction
                    pygame.draw.rect(screen, (0, 0, 0), (0, 50, 350, 350))
                    pygame.draw.rect(screen, (255, 255, 255), (520, 50, 100, 100))

                    pygame.display.flip()

        pygame.display.flip()

