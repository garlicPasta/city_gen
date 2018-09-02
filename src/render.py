import pygame, sys, time
from pygame.locals import *
from pathlib import Path
from drawable import House, Street



class Render:

    UNIT_WIDTH = UNIT_HEIGHT = 16

    def __init__(self, tilemap):
        pygame.init()
        self.tilemap = tilemap
        self.context = {
                'tilemap': tilemap,
                'occupied': [],
                'display': pygame.display.set_mode((800, 600), 0, 32)
                }
        self.house = House()
        self.street = Street()


    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

    def render(self):
        self._handle_events()
        iso_world_cords = [(self.tilemap[x][y] , (x,y))
                for x in range(len(self.tilemap))
                for y in range(len(self.tilemap[0]))]

        for (tile, (x,y)) in iso_world_cords:
            if (x, y) in self.context['occupied']:
                    self.context['occupied'].remove((x, y))
                    continue
            if tile == 0:
                self.house.draw((x,y), self.context)
            if tile == 1:
                self.street.draw((x,y), self.context)
            pygame.display.update()



#constants representing the different resources
E = -1
H = 0
S = 1


#a dictionary linking resources to colours

#a list representing our tilemap
tilemap = [
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, H, H,S, H, H,S, H, H,S, H, H ],
        [ S, S, S,S, S, S,S, S, S,S, S, S ],
        ]


r = Render(tilemap)
r.render()
while True:
    pass
