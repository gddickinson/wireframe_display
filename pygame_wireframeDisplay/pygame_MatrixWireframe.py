#!/bin/env python

import numpy as np
import math
from datetime import datetime
import pyrr
from math import *


class Wireframe:
    def __init__(self, nodes=None):
        self.nodes = np.zeros((0,4))
        self.edges = []
        self.faces = []
        
        self.initiated = False
        self.initTime = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        self.initTime_raw = datetime.now()
        
        self.initialState = []
        self.history = []
        self.historyMax = 20

        self.colour = (255,255,200)

        self.name = ''

        
        if nodes:
            self.addNodes(nodes)      

                      
    def addNodes(self, node_array, reset=False):
        if reset:
           self.nodes = self.nodes = np.zeros((0,4))
        ones_column = np.ones((len(node_array), 1))
        ones_added = np.hstack((node_array, ones_column))
        self.nodes = np.vstack((self.nodes, ones_added))   
        if self.initiated == False:
            self.history.append(self.nodes)
            self.initialState.append(self.nodes)
            self.initiated = True
    
    def addEdges(self, edgeList):
        self.edges += edgeList

    def addFaces(self, face_list, face_colour=(255,255,255), reset=False):
        if reset:
            self.faces = []
            self.edges = []
            
        for node_list in face_list:
            num_nodes = len(node_list)
            if all((node < len(self.nodes) for node in node_list)):
                #self.faces.append([self.nodes[node] for node in node_list])
                self.faces.append((node_list, np.array(face_colour, np.uint8)))
                self.addEdges([(node_list[n-1], node_list[n]) for n in range(num_nodes)])


    def changeShape(self,wireframe):
        center = self.findCentre()
        self.nodes=wireframe.nodes
        self.faces=wireframe.faces
        self.edges=wireframe.edges
        self.moveTo(center[0],center[1],center[2])
        
        
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
        for i, faces in enumerate(self.faces):
            print ("   %d: (%d, %d, %d)" % (i,faces[0][0],faces[0][1],faces[0][2]))

    def output(self):
        if len(self.nodes) > 1:
            self.outputNodes()
        if self.edges:
            self.outputEdges()
        if self.faces:
            self.outputFaces() 

    def nodesToList(self):
        nodeList = []
        for i, (x, y, z, _) in enumerate(self.nodes):
            nodeList.append((x, y, z))  
        return nodeList

    def facesToList(self):
        facesList = []
        for i, faces in enumerate(self.faces):
            facesList.append(faces[0]) 
        return facesList


    # def translate(self, axis, d):
    #     """ Add constant 'd' to the coordinate 'axis' of each node of a wireframe """
            
    #     if axis in ['x', 'y', 'z']:
    #         for node in self.nodes:
    #             setattr(node, axis, getattr(node, axis) + d)

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


    def translationMatrix(self, dx=0, dy=0, dz=0):
        """ Return matrix for translation along vector (dx, dy, dz). """

        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [dx,dy,dz,1]])
    
    def moveTo(self,x,y,z):
        newPos = [x, y, z, 0]
        centerPosition = self.findCentre()
        vector = newPos - centerPosition
        matrix = self.translationMatrix(vector[0], vector[1], vector[2])
        self.update('transform', matrix)      
        
    def shear(self, center, matrix):
        """ Scale the wireframe from the centre of the screen """
    
        for i,node in enumerate(self.nodes):
            self.nodes[i] = center + np.matmul(matrix, node-center)

    def transform(self, matrix):
        """ Apply a transformation defined by a given matrix. """
    
        self.nodes = np.dot(self.nodes, matrix)


    def rotate(self, center, matrix):
        for i, node in enumerate(self.nodes):
            self.nodes[i] = center + np.matmul(matrix, node-center)


    def sortedFaces(self):
        return sorted(self.faces, key=lambda face: min(self.nodes[f][2] for f in face[0]))

    
    def update(self, function, *args):
        """ update call from display """
        #save old nodes
        self.history.append(self.nodes)        
        
        #perform transform
        getattr(self, function)(*args)        
        
        #set limit to number of wireframes
        if len(self.history) == self.historyMax:
            self.history.pop(0)


    def reduceHistory(self):
        if len(self.history) > 0:
            self.history.pop(0)


    def subdivide(self):
        """Subdivide each triangle into four triangles, pushing verts to the unit sphere"""
        
        verts  = self.nodesToList()
        faces = self.facesToList()
        self.nodes = np.zeros((0,4))
        self.faces = []
        self.edges = []
        
        triangles = len(faces)
        for faceIndex in range(triangles):
    
            # Create three new verts at the midpoints of each edge:
            face = faces[faceIndex]
            a, b, c = np.float32([verts[vertIndex] for vertIndex in face])
            verts.append(pyrr.vector.normalize(a + b))
            verts.append(pyrr.vector.normalize(b + c))
            verts.append(pyrr.vector.normalize(a + c))
    
            # Split the current triangle into four smaller triangles:
            i = len(verts) - 3
            j, k = i + 1, i + 2
            faces.append((i, j, k))
            faces.append((face[0], i, k))
            faces.append((i, face[1], j))
            faces[faceIndex] = (k, j, face[2])

            
        self.addNodes(verts)
        self.addFaces(faces)
    
        return 

    def getNodeColour(self):
        return self.colour


# #TRANSFORMATIONS
# def translationMatrix(dx=0, dy=0, dz=0):
#     """ Return matrix for translation along vector (dx, dy, dz). """
    
#     return np.array([[1,0,0,0],
#                      [0,1,0,0],
#                      [0,0,1,0],
#                      [dx,dy,dz,1]])



# def scaleMatrix(sx=0, sy=0, sz=0):
#     """ Return matrix for scaling equally along all axes centred on the point (cx,cy,cz). """
    
#     return np.array([[sx, 0,  0,  0],
#                      [0,  sy, 0,  0],
#                      [0,  0,  sz, 0],
#                      [0,  0,  0,  1]])


# def rotateXMatrix(radians):
#     """ Return matrix for rotating about the x-axis by 'radians' radians """
    
#     c = np.cos(radians)
#     s = np.sin(radians)
#     return np.array([[1, 0, 0, 0],
#                      [0, c,-s, 0],
#                      [0, s, c, 0],
#                      [0, 0, 0, 1]])

# def rotateYMatrix(radians):
#     """ Return matrix for rotating about the y-axis by 'radians' radians """
    
#     c = np.cos(radians)
#     s = np.sin(radians)
#     return np.array([[ c, 0, s, 0],
#                      [ 0, 1, 0, 0],
#                      [-s, 0, c, 0],
#                      [ 0, 0, 0, 1]])

# def rotateZMatrix(radians):
#     """ Return matrix for rotating about the z-axis by 'radians' radians """
    
#     c = np.cos(radians)
#     s = np.sin(radians)
#     return np.array([[c,-s, 0, 0],
#                      [s, c, 0, 0],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, 1]])


# def perspective(c):
#     """Returns the perspective projection"""
#     return np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, -1/c, 1]])    
    
    


if __name__ == "__main__":
    #cube_nodes = [(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    #cube = Wireframe()
    #cube.addNodes(cube_nodes)
    
    #cube.addEdges([(n,n+4) for n in range(0,4)])
    #cube.addEdges([(n,n+1) for n in range(0,8,2)])
    #cube.addEdges([(n,n+2) for n in (0,1,4,5)])
    
    #cube.addFaces([(0,1,3,2), (7,5,4,6), (4,5,1,0), (2,3,7,6), (0,2,6,4), (5,7,3,1)])

    from pygame_basicShapes import *    
    x,y,z = (300,300,250)
    w,h,d = (50,50,50)
    #cube = Cuboid(args1=(x,y,z), args2=(w,h,d))    
    #cube.outputNodes()
    #cube.outputEdges()
    #cube.outputFaces()
    
    tetra = Tetrahedron(args1=(x,y,z), edgeLength=100) 
    tetra.outputNodes()
    tetra.outputEdges()
    tetra.outputFaces()    
    
    
    