import pygame, sys
import numpy as np
from noise import snoise2, pnoise2, snoise3
from pygame.locals import *

#constants representing colours
BLACK = (0,   0,   0  )
BROWN = (153, 76,  0  )
GREEN = (0,   255, 0  )
BLUE  = (0,   0,   255)

#constants representing the different resources
DIRT  = 0
GRASS = 1
WATER = 2
MOUNTAIN  = 3

#a dictionary linking resources to colours
colours =   {
                DIRT  : BROWN,
                GRASS : GREEN,
                WATER : BLUE,
                MOUNTAIN  : BLACK
            }

def draw_map(map, TILESIZE):
    # map.shape[0] = HEIGHT, map.shape[1] = WIDTH
    DISPLAYSURF = pygame.display.set_mode((map.shape[1]*TILESIZE,map.shape[0]*TILESIZE))
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            pygame.draw.rect(
                    DISPLAYSURF,
                    colours[map[y][x]],
                    (x*TILESIZE,y*TILESIZE,TILESIZE,TILESIZE))

def generate_terrain(MAPWIDTH, MAPHEIGHT):
    terrain_map = np.zeros((MAPHEIGHT, MAPWIDTH))
    for y in range(MAPHEIGHT):
        for x in range(MAPWIDTH):
            terrain_map[y][x] = (int(snoise3(y / freq, x / freq, 5, octaves) * 2 + 2))
    return terrain_map

# Remove unnecessary Terrain
def clean_map(map):
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            nbhood = get_neighborhood(map,x,y)
            water = list(nbhood).count(WATER)
            if water >= 7:
                map[y][x] = WATER
# Get a 3x3 grid around the position
# NOTE: doesnt make sense on the border
def get_neighborhood(map, xpos, ypos):
    nbhood = np.zeros((3,3))
    for y in range(nbhood.shape[0]):
        for x in range(nbhood.shape[1]):
            nbhood[y][x] = map[ypos+y-1][xpos+x-1]
    return nbhood

octaves = 10
freq = 10.0 * octaves

map = generate_terrain(100,100)

unique, counts = np.unique(map, return_counts=True)
#dict(zip(unique, counts))
#clean_map(map)
#set up the display
pygame.init()

while True:

    #get all the user events
    for event in pygame.event.get():
        #if the user wants to quit
        if event.type == QUIT:
            #and the game and close the window
            pygame.quit()
            sys.exit()

    draw_map(map, 8)
    #update the display
    pygame.display.update()
