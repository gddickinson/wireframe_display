# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:24:26 2020

@author: g_dic
"""


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams["figure.figsize"] = 12.8, 9.6
import math

def read_obj(filename):
    triangles = []
    vertices = []
    with open(filename) as file:
        for line in file:
            components = line.strip(' \n').split(' ')
            if components[0] == "f": # face data
                # e.g. "f 1/1/1/ 2/2/2 3/3/3 4/4/4 ..."
                indices = list(map(lambda c: int(c.split('/')[0]) - 1, components[1:]))
                for i in range(0, len(indices) - 2):
                    triangles.append(indices[i: i+3])
            elif components[0] == "v": # vertex data
                # e.g. "v  30.2180 89.5757 -76.8089"
                vertex = list(map(lambda c: float(c), components[1:]))
                vertices.append(vertex)
    return np.array(vertices), np.array(triangles)




class Wireframe:
    def __init__(self, nodes=None):
        self.nodes = np.zeros((0,4))
        self.edges = []
        self.faces = []
        
        if nodes:
            self.addNodes(nodes)      
                      
    def addNodes(self, node_array):
        ones_column = np.ones((len(node_array), 1))
        ones_added = np.hstack((node_array, ones_column))
        self.nodes = np.vstack((self.nodes, ones_added))        
    
    
    def addEdges(self, edgeList):
        self.edges += edgeList

    def addFaces(self, face_list, face_colour=(255,255,255)):
        for node_list in face_list:
            num_nodes = len(node_list)
            if all((node < len(self.nodes) for node in node_list)):
                #self.faces.append([self.nodes[node] for node in node_list])
                self.faces.append((node_list, np.array(face_colour, np.uint8)))
                self.addEdges([(node_list[n-1], node_list[n]) for n in range(num_nodes)])


    def outputNodes(self):
        print ("\n --- Nodes --- ")
        for i, (x, y, z, _) in enumerate(self.nodes):
            print ("   %d: (%d, %d, %d)" % (i, x, y, z))

    def outputEdges(self):
        print ("\n --- Edges --- ")
        for i, (node1, node2) in enumerate(self.edges):
            print ("   %d: %d -> %d" % (i, node1, node2))

    def outputFaces(self):
        print ("\n --- Faces --- ")
        for i, nodes in enumerate(self.faces):
            print ("   %d: (%s)" % (i, ", ".join(['%d' % n for n in nodes])))

    def output(self):
        if len(self.nodes) > 1:
            self.outputNodes()
        if self.edges:
            self.outputEdges()
        if self.faces:
            self.outputFaces() 


    def translate(self, axis, d):
        """ Add constant 'd' to the coordinate 'axis' of each node of a wireframe """
            
        if axis in ['x', 'y', 'z']:
            for node in self.nodes:
                setattr(node, axis, getattr(node, axis) + d)

    def scale(self, center, matrix):
        """ Scale the wireframe from the centre of the screen """
    
        for i,node in enumerate(self.nodes):
            self.nodes[i] = center + np.matmul(matrix, node-center)

    def findCentre(self):
        """ Find the centre of the wireframe. """
    
        #num_nodes = len(self.nodes)
        #mean = np.mean(self.nodes) / num_nodes
        mean = self.nodes.mean(axis=0)     # to take the mean of each col
        #print (mean)
        #print "next"
        return mean


    def transform(self, matrix):
        """ Apply a transformation defined by a given matrix. """
    
        self.nodes = np.dot(self.nodes, matrix)


    def rotate(self, center, matrix):
        for i, node in enumerate(self.nodes):
            self.nodes[i] = center + np.matmul(matrix, node-center)


    def sortedFaces(self):
        return sorted(self.faces, key=lambda face: min(self.nodes[f][2] for f in face[0]))

    
    def update(self):
        """ Override this function to control wireframe behaviour. """
        pass

#TRANSFORMATIONS
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


def perspective(c):
    """Returns the perspective projection"""
    return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1/c, 1]])    
    
    

if __name__ == "__main__":
    # cube_nodes = [(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    # cube = Wireframe()
    # cube.addNodes(cube_nodes)
    
    # cube.addEdges([(n,n+4) for n in range(0,4)])
    # cube.addEdges([(n,n+1) for n in range(0,8,2)])
    # cube.addEdges([(n,n+2) for n in (0,1,4,5)])

    # cube.outputNodes()
    # cube.outputEdges()
    # cube.outputFaces()
 
