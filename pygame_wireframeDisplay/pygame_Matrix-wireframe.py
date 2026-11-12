#!/bin/env python

import numpy as np
import math



class Wireframe:
    def __init__(self):
        self.nodes = np.zeros((0, 4))
        self.edges = []
        
                      
    def addNodes(self, node_array):
        ones_column = np.ones((len(node_array), 1))
        ones_added = np.hstack((node_array, ones_column))
        self.nodes = np.vstack((self.nodes, ones_added))        
    
    
    def addEdges(self, edgeList):
        self.edges += edgeList

    def outputNodes(self):
        print ("\n --- Nodes --- ")
        for i, (x, y, z, _) in enumerate(self.nodes):
            print ("   %d: (%d, %d, %d)" % (i, x, y, z))

    def outputEdges(self):
        print ("\n --- Edges --- ")
        for i, (node1, node2) in enumerate(self.edges):
            print ("   %d: %d -> %d" % (i, node1, node2))

    # The methods below came from the object version (node.x, node.y, node.z); they now work on the numpy rows.
    AXES = {'x': 0, 'y': 1, 'z': 2}

    def translate(self, axis, d):
        """ Add constant 'd' to the coordinate 'axis' of each node of a wireframe """
        if axis in self.AXES:
            self.nodes[:, self.AXES[axis]] += d

    def scale(self, centre_x, centre_y, scale):
        """ Scale the wireframe from the centre of the screen """
        self.nodes[:, 0] = centre_x + scale * (self.nodes[:, 0] - centre_x)
        self.nodes[:, 1] = centre_y + scale * (self.nodes[:, 1] - centre_y)
        self.nodes[:, 2] *= scale

    def findCentre(self):
        """ Find the centre of the wireframe. """
        return tuple(self.nodes[:, :3].mean(axis=0))

    def _rotate(self, i, j, radians, centre):
        """ Rotate the (i, j) coordinates about the centre: angle measured from axis j towards axis i, as before. """
        a = self.nodes[:, i] - centre[i]
        b = self.nodes[:, j] - centre[j]
        d = np.hypot(a, b)
        theta = np.arctan2(a, b) + radians
        self.nodes[:, j] = centre[j] + d * np.cos(theta)
        self.nodes[:, i] = centre[i] + d * np.sin(theta)

    def rotateZ(self, radians, args=(0,0,0)):
        self._rotate(1, 0, radians, args)

    def rotateX(self, radians, args=(0,0,0)):
        self._rotate(1, 2, radians, args)

    def rotateY(self, radians, args=(0,0,0)):
        self._rotate(0, 2, radians, args)


    def transform(self, matrix):
        """ Apply a transformation defined by a given matrix. """
    
        self.nodes = np.dot(self.nodes, matrix)


def translationMatrix(dx=0, dy=0, dz=0):
    """ Return matrix for translation along vector (dx, dy, dz). """
    
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [dx,dy,dz,1]])



def scaleMatrix(sx=0, sy=0, sz=0):
    """ Return matrix for scaling equally along all axes centred on the point (cx,cy,cz). """
    
    return np.array([[sx, 0,  0,  0],
                     [0,  sy, 0,  0],
                     [0,  0,  sz, 0],
                     [0,  0,  0,  1]])


def rotateXMatrix(radians):
    """ Return matrix for rotating about the x-axis by 'radians' radians """
    
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rotateYMatrix(radians):
    """ Return matrix for rotating about the y-axis by 'radians' radians """
    
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotateZMatrix(radians):
    """ Return matrix for rotating about the z-axis by 'radians' radians """
    
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])





if __name__ == "__main__":
    cube_nodes = [(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    cube = Wireframe()
    cube.addNodes(cube_nodes)
    
    cube.addEdges([(n,n+4) for n in range(0,4)])
    cube.addEdges([(n,n+1) for n in range(0,8,2)])
    cube.addEdges([(n,n+2) for n in (0,1,4,5)])

    cube.outputNodes()
    cube.outputEdges()
