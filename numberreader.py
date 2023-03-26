import pygame
from pygame.locals import *
import boundingbox
import neuralnetwork

pygame.init()
screen = pygame.display.set_mode((350, 350))
pygame.display.set_caption("Number")

# set background to white
background = pygame.Surface(screen.get_size())
background = background.convert()

# blit background to screen
screen.blit(background, (0, 0))
pygame.display.flip()
draw = False

def save():
    # save the image as number.jpg
    pygame.image.save(screen, "number.jpg")
    boundingbox.boundingbox()
    neuralnetwork.getPred()

while True:
    while True:
        # let the user draw on the screen
        # resolution is 28x28
        # position of the mouse
        m = pygame.mouse.get_pos()
        if (draw):
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
                    # save the image to number.data
                    save()
                    # clear the screen
                    screen.blit(background, (0, 0))
                    pygame.display.flip()
                    break

        pygame.display.flip()

