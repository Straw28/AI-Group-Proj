import pygame
import os

#setting the display size
WIDTH, HEIGHT = 900, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

#Frames per second
FPS = 60 #how fast/ often the while roop refreshes

#Colors
WHITE = (255,255,255)
BLACK = (0, 0, 0)

#setting the grid size:
GRID_SIZE = 100

def draw_window():
    WIN.fill(WHITE) #filling the bg color
    pygame.display.update() #won't update unless we call update
    for i in range(3):
        for j in range(3):
            # Calculate the position of the square
            x = i * GRID_SIZE
            y = j * GRID_SIZE
            # Draw the square
            pygame.draw.rect(screen, WHITE, (x, y, GRID_SIZE, GRID_SIZE), 1)

def mydisplay():  # handles main game loop (redraws window, checks for collisions etc)
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS) #refreshes 60 times per second, we might have tomake this slower tho
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    draw_window()
    pygame.quit()

if __name__ == "__main__":
    mydisplay()
