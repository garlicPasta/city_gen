import pygame
import numpy as np
from random import randint


def _twoDToIso(pos):
    x, y =  pos
    return (16*48 + x - y, 16*16 + (x + y) // 2)

def calc_offset(pos):
    return ( x - y, (x + y) // 2)


class Drawable(object):

    def __init__(self,tile_path, tile_size, unit_size=(1,1)):
        self.tile_size = tile_size
        w,h = self.tile_size
        self.tiles = self._load_tileset(tile_path, w, h)
        self.unit_size = unit_size

    def draw(self, pos, context):
        x_iso, y_iso = _twoDToIso( (pos[0]*16 , pos[1]*16))

        context['to_render'].append((
                self.tiles[randint(0, len(self.tiles)-1)],
                (x_iso, y_iso - self.tile_size[1])
                ))
        context['display'].set_at((x_iso, y_iso) , pygame.Color('red'))

    def _load_tileset(self, filename, width, height):
        image = pygame.image.load(filename).convert()
        image_width, image_height = image.get_size()
        tileset = []
        for tile_y in range(0, image_height//height):
            for tile_x in range(0, image_width//width):
                rect = (tile_x*width, tile_y*height, width, height)
                tile = image.subsurface(rect)
                tile.set_colorkey((255,255,255))
                tileset.append(tile)
        return tileset

    def _calc_occupy(pos):
        x,y = pos
        w,h = self.unit_size
        return [(x+xd, y+yd) for xd in range(1,w) for yd in range(1,h)]

class Grass(Drawable):

    def __init__(self):
        super().__init__('tiles/terrain/grass_0.png', (34, 21))

class Tower(Drawable):

    def __init__(self):
        super().__init__('tiles/skyscrapers_1unit.png', (34, 66))

class Street(Drawable):

    def __init__(self):
        super().__init__('tiles/streets.png', (34, 21))


    def draw(self, pos, context):
        tilemap = context['tilemap']
        x,y = pos
        def isStreet(dx=0, dy=0):
            if 0<= (x+dx) < len(tilemap) and 0<= (y+dy) < len(tilemap[0]):
                return tilemap[x+dx][y+dy] == 'S'
            else:
                return False
        def isStraightX():
            return isStreet(dx=1) and isStreet(dx=-1) and not isStreet(dy=-1) and not isStreet(dy=1)
        def isStraightY():
            return not isStreet(dx=1) and not isStreet(dx=-1) and isStreet(dy=-1) and isStreet(dy=1)

        if isStraightX():
            tile_index = 3
        elif isStraightY():
            tile_index = 2
        else:
            tile_index = 1

        x_iso, y_iso = _twoDToIso( (pos[0]*16 , pos[1]*16))

        context['to_render'].append((
                self.tiles[tile_index],
                (x_iso, y_iso - self.tile_size[1])
                ))

        return tilemap


class House(Drawable):

    def __init__(self):
        super().__init__('tiles/red_brick_buildings.png', (50, 53))

    def draw(self, pos, context):
        assert 'occupied' in context
        assert 'display' in context
        assert 'display' in context

        x,y = pos
        tilemap = context['tilemap']
        if x+1 >= len(tilemap) or tilemap[x+1][y] != 'H':
            return
        context['occupied'].append((x+1,y))

        x_iso, y_iso = _twoDToIso((pos[0] * 16, pos[1] *16))

        context['to_render'].append((
                self.tiles[randint(0, len(self.tiles)-1)],
                ( x_iso, y_iso - self.tile_size[1] + self.unit_size[1] * 8 )
                ))
