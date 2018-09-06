import pygame, sys,os, time, argparse
from pygame.locals import *
from pathlib import Path
from drawable import House, Street, Tower, Grass


class Render:

    UNIT_WIDTH = UNIT_HEIGHT = 16

    def __init__(self, tilemap):
        pygame.init()
        self.tilemap = tilemap
        self.context = {
                'tilemap': tilemap,
                'occupied': [],
                'display': pygame.display.set_mode((1800, 1400), 0, 32),
                'to_render': []
                }
        self.house = House()
        self.street = Street()
        self.tower = Tower()
        self.grass = Grass()


    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

    def renderCity(self):
        self.context['to_render'] = []
        self._handle_events()
        iso_world_cords = [(self.tilemap[x][y] , (x,y))
                for x in range(len(self.tilemap))
                for y in range(len(self.tilemap[0]))]

        for (tile, (x,y)) in iso_world_cords:
            if (x, y) in self.context['occupied']:
                    self.context['occupied'].remove((x, y))
                    continue
            if tile == 'H':
                self.house.draw((x,y), self.context)
            if tile == 'S':
                self.street.draw((x,y), self.context)
            if tile == 'T':
                self.tower.draw((x,y), self.context)
            if tile == 'G':
                self.grass.draw((x,y), self.context)
        self.context['display'].blits(self.context['to_render'])
        pygame.display.update()

    def loop(self):
        while True:
            self._handle_events()

