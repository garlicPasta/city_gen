import pygame, sys
import numpy as np
import scipy as sci
import os
from scipy.interpolate import splprep, splev
import time
from random import randint
from noise import snoise2, pnoise2, snoise3
from pygame.locals import *
import matplotlib.pyplot as plt
import bezier

TILESIZE = 6
WIDTH    = 110
HEIGHT   = 110
citysize = 15

#constants representing colours
RED   = (200,   0,   0  )
LIGHTRED = (255, 100, 80)
GREY  = (125,   125, 125)
BROWN = (153, 76,  0  )
GREEN = (0,   255, 0  )
BLUE  = (0,   0,   255)
BLACK = (0,0,0)

#constants representing the different resources
WATER = 1
GRASS = 2
MOUNTAIN  = 3
HUMAN = 10
STREET = 20
BUILDING = 30

#a dictionary linking resources to colours
colours =   {
                GRASS    : GREEN,
                WATER    : BLUE,
                MOUNTAIN : BROWN,
                HUMAN    : RED,
                STREET   : GREY,
                BUILDING    : LIGHTRED
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
# NOTE:0 doesnt make sense on the border
def get_neighbourhood(map, xpos, ypos):
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
                nbhood = get_neighbourhood(map,x,y)
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

# Create a street from an external point to the citycenter
def generate_mainstreet(map):
    # Generate random external 'mapentrance'
    connected_side = randint(1,4)
    if connected_side % 2 == 1:
        extern = np.array([randint(0, WIDTH), 0])
    else:
        extern = np.array([0, randint(0, HEIGHT)])

    # Create the route from the entrance to the citycentrum
    distance = np.linalg.norm(extern-np.array([citycenter[1], citycenter[0]]))
    sections = int(round(distance/10))
    # Create a point every ~10 tiles with a small random noise
    # These are the control points for a beziercurve forming the street
    points = [extern]
    for i in range(1,sections):
        # points += linesegment + random noise and correction so that the points do not land out of boundary
        points += [(((1-i/sections)*extern+i/sections*citycenter).astype(int) + np.random.randint(-4,4,size=2)) / np.max([WIDTH,HEIGHT])*(np.max([WIDTH,HEIGHT])-8) + 4]
    points += [citycenter]

    # Create a beziercurve with points as control points
    curve = bezier.Curve(np.asfortranarray(points, dtype = float).T, degree=2)
    for i in range(2*int(1.1*curve.length)):
        pt = curve.evaluate(i/(2*curve.length)).astype(int) # Evaluate gleichverteilt über die kurve
        map[pt[1][0]][pt[0][0]] = STREET

# Fill up the diagonal pieces of the street
def fix_mainstreet(map):
    for y in range(1,map.shape[0]-1):
        for x in range(1,map.shape[1]-1):
            # Check if street
            if map[y][x] == STREET:
                # check if the street is diagonal
                if map[y-1][x-1] == STREET and map[y][x-1] != STREET:
                    map[y-1][x] = STREET # Fill the diagonal
                    #map[y-1][x+1] = GRASS# Create grass in the surrounding
                elif map[y-1,x+1] == STREET and map[y][x+1] != STREET:
                    map[y-1][x] = STREET

# A quadratic matrix of size 2r+1  with a circle of radius r of ones around the center
def circle_matrix(r):
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    return (dists < r+0.5).astype(int)

# Assign a score to a position on the map
def score_location(map, xpos, ypos):
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
    zielwasser = desired_water*np.pi*citysize**2
    score += 2*(zielwasser-abs(counter[WATER]-zielwasser))
    return score

# Assign a score value to each buildable tile (GRASS)
def score_locations(map):
    scores = np.zeros(map.shape)
    for y in range(citysize, map.shape[0]-citysize):
        for x in range(citysize, map.shape[1]-citysize):
            # check if buildable
            if map[y][x] == GRASS:
                scores[y][x] = score_location(map, x, y)
            else:
                scores[y][x] = 0
    return scores

# range is the influence range of streets and should influence the sparseness of the city
def score_street(map, r, xpos, ypos, street_density, citycenter):
    # street_density = Desired amount of streettiles in the given radius / density
    map_chunk = map[ypos-r:ypos+r+1,xpos-r:xpos+r+1] # The selected part of the map
    mask = np.multiply(circle_matrix(r), map_chunk) # only the circle
    counter = dict.fromkeys(colours)
    for TILETYPE in colours:
        counter[TILETYPE] = np.sum(mask==TILETYPE)
    score = 0
    zielstrasse = street_density*np.pi*r**2
    score += zielstrasse-abs(counter[STREET]-zielstrasse) # es soll eine gewisse Straßendichte erreicht werden
    score += 0.1*counter[GRASS] # Auch genug Baufläche muss vorhanden sein
    score += citysize - np.linalg.norm(citycenter-np.array([xpos, ypos])) # Build near the citycenter in the given radius
    return score

# Assign each tile the 'buildability'/how nice it is to build there
def score_streets(map, r, street_density, citycenter):
    scores = np.zeros(map.shape)
    for y in range(r,map.shape[0]-r):
        for x in range(r,map.shape[1]-r):
            # check if buildable
            if map[y][x] == GRASS:
                scores[y][x] = score_street(map,r,x,y, street_density, citycenter)
            else:
                scores[y][x] = 0
    return scores

# Assign a score telling the HouseBuilders where to go
def score_building(map, building_range, xpos, ypos):
    map_chunk = map[ypos-building_range:ypos+building_range+1,xpos-building_range:xpos+building_range+1] # The selected part of the map
    mask = np.multiply(circle_matrix(building_range), map_chunk) # only the circle
    counter = dict.fromkeys(colours)
    for TILETYPE in colours:
        counter[TILETYPE] = np.sum(mask==TILETYPE)
    score = 0
    score += counter[WATER] # Building near water is preferred
    score += counter[STREET] # You need to build along streets
    score += counter[GRASS] # Auch genug Baufläche muss vorhanden sein
    score -= counter[BUILDING] # Es ist attraktiver an stellen zu bauen, wo mehr Fläche vorhanden ist
    #score += citysize - np.linalg.norm(citycenter-np.array([xpos, ypos])) # Build near the citycenter in the given radius
    return score

# Assign a housebuilder core to each tile
def score_buildings(map, building_range = 2):
    scores = np.zeros(map.shape)
    for y in range(building_range, map.shape[0]-building_range):
        for x in range(building_range, map.shape[1]-building_range):
            # check if buildable
            if map[y][x] == STREET:
                scores[y][x] = score_building(map, building_range, x, y)
            else:
                scores[y][x] = 0
    return scores

# Is it okay to build a street at the given position
def street_allowed(map, xpos, ypos):
    if map[ypos][xpos] == GRASS:
        buildable = True # Is this tile allowd to be a street
        # Vier 2x2 quadrate um die Position herum abtasten
        quadrate = []
        for i in range(4): # left to right, top to bottom
            quadrate += [np.array([[map[ypos-1+(i>1), xpos-1+i%2], map[ypos-1+(i>1), xpos+i%2]],
                                   [map[ypos+(i>1), xpos-1+i%2], map[ypos+(i>1), xpos+i%2]]])]
        for q in quadrate:
            if np.sum(q == STREET) > 2:
                buildable = False
    else:
        buildable = False
    return buildable

class Agent:
    def __init__(self, startx, starty):
        self.x = startx
        self.y = starty
        self.type = 'agent'
        self.alive = True
    # Intercat with the enviroment
    def changeEnv(self, map):
        pass
    # change position
    def move(self, map, cityscore):
        pass
    def __str__(self):
        return self.type + ' at [' + str(self.x) + ', ' + str(self.y) + ']'

# HouseBuilder -> bis zu abstand zwei von straße bauen
# StreetBuilder -> im abstand eins von Häusern score erhöhen
class HouseBuilder(Agent):
    def __init__(self, startx, starty):
        super().__init__(startx, starty)
        self.type = 'housebuilder'

    def changeEnv(self, map):
        # is grass nearby? if so build a house
        if map[self.y][self.x] == GRASS:
            # TODO: Buildable function
                map[self.y-1][self.x] = BUILDING # build houses

    def move(self, map, cityscore):
        directions = [map[self.y-1][self.x], # oben
                      map[self.y+1][self.x], # unten
                      map[self.y][self.x-1], # links
                      map[self.y][self.x+1]] # rechts
        # can only move on streets



class StreetBuilder(Agent):
    def __init__(self, startx, starty):
        super().__init__(startx, starty)
        self.type = 'streetbuilder'

    # Ziel: maximiere platz für Häuser/bebaubarkeit
    def changeEnv(self, map):
        if map[self.y][self.x] == GRASS:
            if street_allowed(map, self.x, self.y):
                map[self.y][self.x] = STREET # build street

    def move(self, map, cityscore):
        if self.x == WIDTH-1 or self.x == 1 or self.y == HEIGHT-1 or self.y == 1:
            self.alive = False
            print(str(self) + ' died.')
        directions = [cityscore[self.y-1][self.x], # oben
                      cityscore[self.y+1][self.x], # unten
                      cityscore[self.y][self.x-1], # links
                      cityscore[self.y][self.x+1]] # rechts
        # TODO: Rule out
        # Rule out the positions where we are not allowed to build
        for i in range(4):
            if not(street_allowed(map, self.x+(2*(i%2)-1)*(i>1), self.y+(2*(i%2)-1)*(i<2) )):
                directions[i] = 0

        sorted = np.sort(directions)
        goto = randint(1,10)
        if goto > 3:
            chosen = sorted[3]
        else:
            chosen = sorted[2]
        direct = directions.index(chosen)# Vier 2x2 quadrate um die Position herum abtasten

        if direct == 0: #and map[self.y-1][self.x] == GRASS:
            self.y -= 1
        elif direct == 1: #and map[self.y+1][self.x] == GRASS:
            self.y += 1
        elif direct == 2: #and map[self.y][self.x-1] == GRASS:
            self.x -= 1
        elif direct == 3: #and map[self.y][self.x+1] == GRASS:
            self.x += 1


# Terrain parameters
octaves = 6
freq = 10.0 * octaves

map = generate_terrain(WIDTH,HEIGHT)
# TODO:10 erode and dilate
clean_map(map) # removes bad terrain

scoremap = score_locations(map)

# Find maximum score
max = np.unravel_index(np.argmax(scoremap, axis=None), scoremap.shape)
citycenter = np.array([max[1],max[0]])
map[max[0]][max[1]] = STREET

generate_mainstreet(map)
fix_mainstreet(map)

street_density = 0.1
number_street_builder = 5 # Number of outgoing streets/ starting agents
number_house_builder = 5 # Number of agents that build houses

agents = []
for i in range(number_street_builder):
    agents += [StreetBuilder(citycenter[0], citycenter[1])]
for i in range(number_house_builder):
    agents += [HouseBuilder(citycenter[0], citycenter[1])]


cityscore = score_streets(map, 5, street_density, citycenter)
builderscore = score_buildings(map)

# Plot scores
plt.rcParams['figure.figsize'] = [10, 10]
#plt.matshow(scoremap, interpolation='nearest', cmap = 'jet')
#plt.matshow(cityscore, interpolation='nearest', cmap = 'Blues')
plt.matshow(builderscore, interpolation='nearest', cmap = 'jet')
plt.colorbar()
plt.show()

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
            BLACK,
            (citycenter[0]*TILESIZE,citycenter[1]*TILESIZE,TILESIZE,TILESIZE))

    for agent in agents:
        if agent.alive:
            agent.move(map, cityscore)
            agent.changeEnv(map)
            pygame.draw.rect(
                    DISPLAYSURF,
                    RED,
                    (agent.x*TILESIZE,agent.y*TILESIZE,TILESIZE,TILESIZE))
    cityscore = score_streets(map, 5, street_density, citycenter)
    time.sleep(.1)
    #update the display
    pygame.display.update()
