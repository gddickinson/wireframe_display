"""Pygame-free scene logic of the matrix wireframe viewer: transform matrices, scene-wide operations and key table.

ProjectionViewer (pygame_MatrixDisplay.py) subclasses WireframeScene and adds only the Pygame window, event loop and
drawing. The web viewer (web/static/wire_engine.js) is a port of this module and is tested against it.

Conventions (unchanged from the original viewer): nodes are rows (x, y, z, 1) in screen pixels, y pointing down.
translationMatrix is applied as row vectors (nodes @ M); rotation, scale and shear matrices are applied about a centre
c as c + M @ (node - c). The projection is orthographic: a node is drawn at its (x, y).
"""
import math
from random import randint

import numpy as np

# (pygame key name, ProjectionViewer method, arguments, description). pygame_MatrixDisplay builds its key_to_function
# from this table; the web viewer uses the same table (exported by the web API).
KEY_BINDINGS = [
    ("LEFT", "translateAll", [[-10, 0, 0]], "move left"),
    ("RIGHT", "translateAll", [[10, 0, 0]], "move right"),
    ("DOWN", "translateAll", [[0, 10, 0]], "move down"),
    ("UP", "translateAll", [[0, -10, 0]], "move up"),
    ("EQUALS", "scaleAll", [1.05], "zoom in (scale about the screen centre)"),
    ("MINUS", "scaleAll", [0.95], "zoom out"),
    ("q", "rotateAll", ["X", 0.1], "rotate about X +"),
    ("w", "rotateAll", ["X", -0.1], "rotate about X -"),
    ("a", "rotateAll", ["Y", 0.1], "rotate about Y +"),
    ("s", "rotateAll", ["Y", -0.1], "rotate about Y -"),
    ("z", "rotateAll", ["Z", 0.1], "rotate about Z +"),
    ("x", "rotateAll", ["Z", -0.1], "rotate about Z -"),
    ("e", "toggleEdges", [], "show / hide edges"),
    ("d", "toggleNodes", [], "show / hide nodes"),
    ("t", "shearAll", [[-0.1, 0, 0, 0, 0, 0]], "shear xy -"),
    ("y", "shearAll", [[0.1, 0, 0, 0, 0, 0]], "shear xy +"),
    ("g", "shearAll", [[0, -0.1, 0, 0, 0, 0]], "shear xz -"),
    ("h", "shearAll", [[0, 0.1, 0, 0, 0, 0]], "shear xz +"),
    ("b", "shearAll", [[0, 0, -0.1, 0, 0, 0]], "shear yz -"),
    ("n", "shearAll", [[0, 0, 0.1, 0, 0, 0]], "shear yz +"),
    ("m", "subdivideAll", [], "subdivide triangles"),
    ("0", "centerAll", [], "centre on screen"),
    ("9", "changeAllShapes", [], "change every shape to the tetrahedron"),
]


# --- transform matrices ---------------------------------------------------------------------------------------------
def perspectiveMatrix(c):
    """Returns the perspective projection (defined in the original viewer, not used for drawing)."""
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, -1 / c, 1]])


def shearMatrix(hxy=0, hxz=0, hyz=0, hyx=0, hzx=0, hzy=0):
    """ Return matrix for shear along vector (hxy, hxz, hyz, hyx, hzx, hzy). """
    return np.array([[1, hxy, hxz, 0], [hyx, 1, hyz, 0], [hzx, hzy, 1, 0], [0, 0, 0, 1]])


def translationMatrix(dx=0, dy=0, dz=0):
    """ Return matrix for translation along vector (dx, dy, dz). """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [dx, dy, dz, 1]])


def scaleMatrix(sx=0, sy=0, sz=0):
    """ Return matrix for scaling along the axes (applied about a centre by Wireframe.scale). """
    return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])


def rotateXMatrix(radians):
    """ Return matrix for rotating about the x-axis by 'radians' radians """
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def rotateYMatrix(radians):
    """ Return matrix for rotating about the y-axis by 'radians' radians """
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rotateZMatrix(radians):
    """ Return matrix for rotating about the z-axis by 'radians' radians """
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def identityMatrix():
    return np.eye(4, dtype=int)


def project(wireframe):
    """Screen positions of the nodes: the viewer's orthographic projection (x, y of each node)."""
    return wireframe.nodes[:, :2].copy()


class WireframeScene:
    """Named wireframes on a width x height screen, with the viewer's scene-wide operations (no Pygame)."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.wireframes = {}
        self.displayNodes = True
        self.displayEdges = True
        self.nodeColour = (255, 255, 255)
        self.edgeColour = (200, 200, 200)
        self.nodeRadius = 4
        self.animate = False
        self.updateEvent = False
        self.randomEvent = False
        self.showHistory = False
        self.colourShift = False
        self.useWireframeColour = False
        self.useRandomColour = False
        self.movePath_n = 200
        self.moveIndex = 0
        self.moveCenter = (self.width / 2, self.height / 2, 0)
        self.history = []
        self.yoyoCounter = 0

    # matrices as methods, as in the original ProjectionViewer
    perspectiveMatrix = staticmethod(perspectiveMatrix)
    shearMatrix = staticmethod(shearMatrix)
    translationMatrix = staticmethod(translationMatrix)
    scaleMatrix = staticmethod(scaleMatrix)
    rotateXMatrix = staticmethod(rotateXMatrix)
    rotateYMatrix = staticmethod(rotateYMatrix)
    rotateZMatrix = staticmethod(rotateZMatrix)
    identityMatrix = staticmethod(identityMatrix)

    def addWireframe(self, name, wireframe):
        """ Add a named wireframe object. """
        self.wireframes[name] = wireframe
        self.history.append(name)

    def removeWireframe(self, name):
        try:
            self.wireframes.pop(name)
            self.history.remove(name)
        except Exception:
            print(name, ' not found')

    def getRandomColour(self):
        return (randint(0, 255), randint(0, 255), randint(0, 255))

    def shiftColour(self, colour, count):
        r = max(colour[0] - count, 0)
        g = max(colour[1] - 2 * count, 0)
        b = max(colour[2] - 5 * count, 0)
        return (r, g, b, 1 / (count + 1))

    def shearAll(self, vector):
        """ Shear all wireframes (symmetric shear: hxy = hyx, hxz = hzx, hyz = hzy). """
        matrix = self.shearMatrix(vector[0], vector[1], vector[2], vector[0], vector[1], vector[2])
        for wireframe in self.wireframes.values():
            center = wireframe.findCentre()
            wireframe.update('shear', center, matrix)

    def translateAll(self, vector):
        """ Translate all wireframes along a given axis by d units. """
        matrix = self.translationMatrix(vector[0], vector[1], vector[2])
        self.moveCenter = (self.moveCenter[0] + vector[0], self.moveCenter[1] + vector[1], self.moveCenter[2] + vector[2])
        for wireframe in self.wireframes.values():
            wireframe.update('transform', matrix)

    def scaleAll(self, scale):
        """ Scale all wireframes by a given scale, centred on the centre of the screen. """
        center = [self.width / 2, self.height / 2, 0, 0]
        matrix = self.scaleMatrix(scale, scale, scale)
        for wireframe in self.wireframes.values():
            wireframe.update('scale', center, matrix)

    def rotateAll(self, axis, theta):
        """ Rotate all wireframe about their centre, along a given axis by a given angle. """
        rotateFunction = 'rotate' + axis + 'Matrix'
        for wireframe in self.wireframes.values():
            center = wireframe.findCentre()
            matrix = getattr(self, rotateFunction)(theta)
            wireframe.update('rotate', center, matrix)

    def centerAll(self):
        center = [self.width / 2, self.height / 2, 0, 0]
        for wireframe in self.wireframes.values():
            vector = center - wireframe.findCentre()
            wireframe.update('transform', self.translationMatrix(vector[0], vector[1], vector[2]))

    def moveAll(self, x, y, z):
        newPos = [x, y, z, 0]
        for wireframe in self.wireframes.values():
            vector = newPos - wireframe.findCentre()
            wireframe.update('transform', self.translationMatrix(vector[0], vector[1], vector[2]))

    def subdivideAll(self):
        for wireframe in self.wireframes.values():
            wireframe.subdivide()

    def toggleEdges(self):
        self.displayEdges = not self.displayEdges

    def toggleNodes(self):
        self.displayNodes = not self.displayNodes

    def randomColour(self):
        self.nodeColour = self.getRandomColour()

    def changeAllShapes(self, shape=None):
        if shape is None:
            from pygame_basicShapes import Tetrahedron
            shape = Tetrahedron(args1=(0, 0, 0), edgeLength=100)
        for wireframe in self.wireframes.values():
            wireframe.changeShape(shape)

    def animate_update(self):
        self.spin()
        self.yoyo()

    def randomMove(self):
        self.moveAll(randint(0, self.width), randint(0, self.height), randint(0, 100))

    def pointsInCircum(self, r, n=100):
        update = self.moveIndex % n
        return [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)][update]

    def moveAroundShape(self, pts):
        self.moveAll(pts[0] + self.moveCenter[0], pts[1] + self.moveCenter[1], self.moveCenter[2])
        self.moveIndex = self.moveIndex + 1
        if self.moveIndex > self.movePath_n:
            self.moveIndex = 0

    def zMotion(self, direction='away'):
        self.scaleAll(0.85 if direction == 'away' else 1.1762)

    def yoyo(self):
        self.zMotion() if self.yoyoCounter < 20 else self.zMotion(direction='towards')
        self.yoyoCounter += 1
        if self.yoyoCounter >= 40:
            self.yoyoCounter = 0

    def spin(self):
        self.translateAll([-10, 0, 0])
        self.rotateAll('Z', 0.1)
        self.translateAll([10, 0, 0])
        self.rotateAll('X', 0.1)
        self.translateAll([0, 10, 0])
        self.rotateAll('Y', 0.1)
        self.translateAll([0, -10, 0])

    def update(self):
        for wireframe in self.wireframes.values():
            wireframe.reduceHistory()

    def random_update(self):
        self.randomColour()

    def apply_key(self, key):
        """Run the operation bound to a key name of KEY_BINDINGS (e.g. 'q', 'LEFT'). Returns False if unbound."""
        for name, method, args, _ in KEY_BINDINGS:
            if name == key:
                getattr(self, method)(*args)
                return True
        return False
