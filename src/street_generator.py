from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import defaultdict
from render import Render

mean = [0, 0]
cov = [[1.8, 0], [0, 1.8]]

POLERADIUS = 2
DISTRICTS = 80
CITY_SIZE = (80,80)
POLEPOINTS = 5

class District:

    def __init__(self):
        self.cords = []
        self.tiles = []

    def add_tile(self, t):
        self.cords.append(t)

    def evaluate_tiles(self, poles):
        matrix = np.array(self.tiles)
        x, y = np.random.multivariate_normal([0,0], cov, len(self.cords)).T * POLERADIUS
        rands = np.array(list(zip(x,y)))
        i = 0
        for cords in self.cords:
            tile = "H"
            for pole in poles:
                dist = np.linalg.norm(pole - np.array(cords))
                if dist < np.linalg.norm(rands[i]) and tile == "H":
                    tile = "T"
            self.tiles.append((tile, cords))
            i +=1

    def calcCentroid(self):
        matrix = np.array(self.tiles)


class DistrictBuilder:

    def build(self, streetmap, polepoints):
        im2, contours, hierarchy = cv2.findContours(
                streetmap.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE
                )

        districtmap = np.zeros(streetmap.shape)
        for i in range(len(contours)-1):
            cv2.drawContours(districtmap, contours, i, i+2, cv2.FILLED)

        districtmap[streetmap.astype(bool)] = 0
        plt.imshow(districtmap)
        plt.show()
        districts = defaultdict(lambda: District())
        it = np.nditer(districtmap, flags=['multi_index'])
        while not it.finished:
            if int(it[0]) != 0:
                d = districts[int(it[0])]
                d.add_tile(it.multi_index)
            it.iternext()

        for key in districts:
            districts[key].evaluate_tiles(polepoints)
        return districts


class StreetGenerator:

    def __init__(self, width, length):
        self.width = width
        self.length = length
        self.streetmap = np.zeros((width, length))
        self.districtmap = np.zeros((width, length))

    def genUniformPoints(self):
        return np.random.uniform(0, max(self.width, self.length), DISTRICTS).reshape(-1,2)

    def genGaussPoints(self):
        x, y = np.random.multivariate_normal(mean, cov, 100).T
        return np.array(list(zip(x,y)))

    def _mapStreetsToTile(self, number):
        if number>=1:
            return 'S'
        else:
            return 'H'

    def _mapStreetsToBinary(self, number):
        if number<50:
            return 1
        else:
            return 0

    def set(self, x,y):
        self.streetmap[y,x] = 1

    def line(self, p1,p2):
        "Bresenham's line algorithm"

        def set(x,y):
            self.streetmap[x,y] = 1

        x0, y0 = p1
        x1, y1 = p2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                set(x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                set(x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        set(x, y)

    def genGrid(self,size, rectsize):
        grid = np.zeros(size, dtype='int64')
        for x in range(size[0]):
            for y in range(size[1]):
                if x % rectsize == 0 :
                    grid[x,y] = 1
                if y % rectsize == 0 :
                    grid[x,y] = 1
        return grid

    def normPoints(self, points):
        points -= np.min(points)
        points = points / np.max(np.abs(points))
        points[:,1] *= self.width-1
        points[:,0] *= self.length-1
        return points.astype(int)

    def drawVoronoi(self, verts, voro):
        for edge in voro.ridge_vertices:
            if not(-1 in edge):
                p1_index,p2_index = edge
                p1 = verts[p1_index]
                p2 = verts[p2_index]
                self.line(p1,p2)

    def filterPointsByBoundingBox(self, ps, bb):
        x_min, x_max = bb[0,:]
        y_min, y_max = bb[1,:]

        def insideBB(p):
            x,y = p
            x_min, x_max = bb[0,:]
            y_min, y_max = bb[1,:]
            return x_min <= x <=x_max and y_min <= y <=y_max


        def handleOutSidePoint(p):
            x,y = p
            if x_min <= x <=x_max :
                if y< y_min:
                    return np.array([x, y_min])
                else:
                    return np.array([x, y_max])
            elif y_min <= y <=y_max :
                if x< x_min:
                    return np.array([x_min, y])
                else:
                    return np.array([x_max, y])
            corners = np.array([
                    [x_min, y_min],
                    [x_min, y_max],
                    [x_max, y_min],
                    [x_max, y_max]
                    ])
            index = np.argmin(np.linalg.norm(corners - p,axis=1))
            return corners[index]

        f_p = []
        i = 0
        for p in ps:
            if insideBB(p):
                f_p.append(p)
                i+=1
            else:
                f_p.append(handleOutSidePoint(p))

        return f_p

    def genStreetTileMap(self):
        random_points = self.genUniformPoints()
        vor = Voronoi(random_points)
        points = vor.points
        verts = vor.vertices
        bbox = np.array([
            [np.min( points[:,0]), np.max(points[:,0])],
            [np.min( points[:,1]), np.max(points[:,1])]
            ])
        f_p = self.filterPointsByBoundingBox(verts, bbox)
        verts_norm = self.normPoints(f_p)
        self.drawVoronoi(verts_norm, vor)
        np.random.shuffle(verts_norm)
        return self.streetmap, verts_norm[:POLEPOINTS]

class CityBuilder:

    def __init__(self, width, length):
        self.width = width
        self.length = length
        self.sg = StreetGenerator(width, length)
        self.db = DistrictBuilder()

    def build(self):
        streetmap, polepoints = self.sg.genStreetTileMap()
        districts = self.db.build(streetmap, polepoints)
        city = np.empty((self.width, self.length), dtype='U')
        city[:] = 'G'
        for key in districts:
            district = districts[key]
            for tile in district.tiles:
                t,(x,y) = tile
                city[x,y] = t
        city[streetmap.astype(bool)] = 'S'
        return city

w,l = CITY_SIZE
city_builder = CityBuilder(w, l)
tilemap = city_builder.build()
r = Render(tilemap)
r.renderCity()

while(True):
    r.loop()
