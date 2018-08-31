import pygame, sys
import numpy as np
import scipy
from random import randint
from noise import snoise2, pnoise2, snoise3
from pygame.locals import *

TILESIZE = 8

#constants representing colours
RED   = (200,   0,   0  )
BROWN = (153, 76,  0  )
GREEN = (0,   255, 0  )
BLUE  = (0,   0,   255)

#constants representing the different resources
WATER = 1
GRASS = 2
MOUNTAIN  = 3
HUMAN = 10
#a dictionary linking resources to colours
colours =   {
                GRASS    : GREEN,
                WATER    : BLUE,
                MOUNTAIN : BROWN,
                HUMAN    : RED
            }

def draw_map(map, DISPLAYSURF, TILESIZE):
    # map.shape[0] = HEIGHT, map.shape[1] = WIDTH
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            pygame.draw.rect(
                    DISPLAYSURF,
                    colours[map[y][x]],
                    (x*TILESIZE,y*TILESIZE,TILESIZE,TILESIZE))

def generate_terrain(MAPWIDTH, MAPHEIGHT):
    terrain_map = np.zeros((MAPHEIGHT, MAPWIDTH))
    seed = randint(0,1000)
    for y in range(MAPHEIGHT):
        for x in range(MAPWIDTH):
            terrain_map[y][x] = round(snoise3(y / freq, x / freq, seed, octaves)+2)
    return terrain_map

# Get a 3x3 grid around the position
# NOTE: doesnt make sense on the border
def get_neighborhood(map, xpos, ypos):
    nbhood = np.zeros((3,3))
    for y in range(nbhood.shape[0]):
        for x in range(nbhood.shape[1]):
            nbhood[y][x] = map[ypos+y-1][xpos+x-1]
    return nbhood

# Remove unnecessary Terrain
def clean_map(map):
    changed = True
    while changed == True:
        for y in range(1,map.shape[0]-1):
            for x in range(1,map.shape[1]-1):
                nbhood = get_neighborhood(map,x,y)
                if (map[y][x] == GRASS) and (np.count_nonzero(nbhood == WATER) >= 7):
                    map[y][x] = WATER
                    changed = True
                elif (map[y][x] == WATER) and (np.count_nonzero(nbhood == GRASS) >= 7):
                    map[y][x] = GRASS
                    changed = True
                elif (map[y][x] == MOUNTAIN) and (np.count_nonzero(nbhood == GRASS) >= 7):
                    map[y][x] = GRASS
                    changed = True
                else:
                    changed = False

# A quadratic matrix of size 2r+1  with a circle of radius r of ones around the center
def circle_matrix(r):
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    return (dists < r+0.5).astype(int)
print(circle_matrix(4))


def score_location(map, citysize, xpos, ypos):
    map_chunk = map[ypos-citysize:ypos+citysize+1,xpos-citysize:xpos+citysize+1] # The selected part of the map
    mask = np.multiply(circle_matrix(citysize), map_chunk) # only the circle
    desired_water = 0.1 # How much water is desired

    counter = dict.fromkeys(colours)
    for TERRAIN in colours:
        counter[TERRAIN] = np.sum(mask==TERRAIN)
    # assign a score to the counter
    score = 0
    score += counter[GRASS]
    score -= counter[MOUNTAIN]
    score += np.log(counter[WATER]/(np.pi*citysize**2)-desired_water)
    return score
print(score_location(map, 10, 50,50))

# Assign a score value to each buildable tile (GRASS)
def score_locations(map,citysize):
    scores = np.zeros(map.shape)
    for y in range(citysize,map.shape[0]-citysize):
        for x in range(citysize,map.shape[1]-citysize):
            # check if buildable
            if map[y][x] == GRASS:
                scores[y][x] = score_location(map,citysize,x,y)
            else:
                scores[y][x] = 0
    return scores

octaves = 6
freq = 10.0 * octaves

map = generate_terrain(100,100)


# NOTE: erode and dilate
clean_map(map) # removes bad terrain

scoremap = score_locations(map,15)
maxel = np.unravel_index(np.argmax(scoremap, axis=None), scoremap.shape)
print(maxel[0])
#set up the display
pygame.init()

DISPLAYSURF = pygame.display.set_mode((map.shape[1]*TILESIZE,map.shape[0]*TILESIZE))

while True:
    #get all the user events
    for event in pygame.event.get():
        #if the user wants to quit
        if event.type == QUIT:
            #and the game and close the window
            pygame.quit()
            sys.exit()

    draw_map(map, DISPLAYSURF, TILESIZE)
    pygame.draw.rect(
            DISPLAYSURF,
            RED,
            (maxel[1]*TILESIZE,maxel[0]*TILESIZE,TILESIZE,TILESIZE))
    #update the display
    pygame.display.update()
