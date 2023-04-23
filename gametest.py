import pygame
import os

#setting the display size
WIDTH, HEIGHT = 900, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning and SARSA Visualization")


#Frames per second
FPS = 60 #how fast/ often the while roop refreshes
VELOCITY = 5

#loading images
YELLOW_SPACESHIP_IMAGE = pygame.image.load('spaceship_yellow.png')
RED_SPACESHIP_IMAGE = pygame.image.load('spaceship_red.png')

#resizing image
SPACESHIP_WIDTH, SPACESHIP_HEIGHT = 55, 40
YELLOW_SPACESHIP = pygame.transform.scale(YELLOW_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT))
RED_SPACESHIP = pygame.transform.scale(RED_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT))


#rotating spaceship
YELLOW_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(YELLOW_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)), 90) #object we're rotating and angle of rotation
RED_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(RED_SPACESHIP_IMAGE, (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)), 270)

#Colors
WHITE = (255,255,255)
BLACK = (0, 0, 0)

#setting the grid size:
GRID_SIZE = 100

def draw_window(red, yellow):
    WIN.fill(WHITE) #filling the bg color
    WIN.blit(YELLOW_SPACESHIP,(yellow.x, yellow.y)) #when you wanna draw a surface onto a screen
    WIN.blit(RED_SPACESHIP, (red.x, red.y))              # the reason we access to x and y is bc we're using a rectangle to represent the spaceship. predefines the x and y as whatever we set it to 
    pygame.display.update() #won't update unless we call update


def main():  # handles main game loop (redraws window, checks for collisions etc)
    red = pygame.Rect(700, 300, SPACESHIP_WIDTH, SPACESHIP_HEIGHT) # x, y , width, height (of thingy)
    yellow = pygame.Rect(100, 300, SPACESHIP_WIDTH, SPACESHIP_HEIGHT) # x, y , width, height (of thingy)
    
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS) #refreshes 60 times per second, we might have tomake this slower tho
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        keys_pressed = pygame.key.get_pressed() #every single time loop runs, tells us what keys are currently being pressed down, if the ones we're looking for are pressed, respond
        if keys_pressed[pygame.K_a]: #LEFT
            yellow.x -= VELOCITY
        if keys_pressed[pygame.K_d]: #RIGHT
            yellow.x += VELOCITY
        if keys_pressed[pygame.K_w]: #UP
            yellow.y -= VELOCITY
        if keys_pressed[pygame.K_s]: #DOWN
            yellow.y += VELOCITY

        draw_window(red, yellow)
    pygame.quit()

if __name__ == "__main__":
    main()
