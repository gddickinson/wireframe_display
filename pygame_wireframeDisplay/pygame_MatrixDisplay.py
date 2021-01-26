import pygame_MatrixWireframe as wf
import pygame, sys
import numpy as np
from pygame_basicShapes import *
from pygame_parametricShapes import *
import time
from random import randint
import math
from datetime import datetime, timedelta

# matrix transforms
key_to_function = {
    pygame.K_LEFT: (lambda x: x.translateAll([-10, 0, 0])),
    pygame.K_RIGHT:(lambda x: x.translateAll([ 10, 0, 0])),
    pygame.K_DOWN: (lambda x: x.translateAll([0,  10, 0])),
    pygame.K_UP:   (lambda x: x.translateAll([0, -10, 0])),
	
    pygame.K_EQUALS: (lambda x: x.scaleAll(1.05)),
    pygame.K_MINUS:  (lambda x: x.scaleAll( 0.95)),
    
    pygame.K_q:      (lambda x: x.rotateAll('X',  0.1)),
    pygame.K_w:      (lambda x: x.rotateAll('X', -0.1)),
    pygame.K_a:      (lambda x: x.rotateAll('Y',  0.1)),
    pygame.K_s:      (lambda x: x.rotateAll('Y', -0.1)),
    pygame.K_z:      (lambda x: x.rotateAll('Z',  0.1)),
    pygame.K_x:      (lambda x: x.rotateAll('Z', -0.1)),
    
    pygame.K_e:      (lambda x: x.toggleEdges()),
    pygame.K_d:      (lambda x: x.toggleNodes()),

    pygame.K_t:      (lambda x: x.shearAll([-0.1, 0, 0, 0, 0, 0])),
    pygame.K_y:      (lambda x: x.shearAll([0.1, 0, 0, 0, 0, 0])),
    pygame.K_g:      (lambda x: x.shearAll([0, -0.1, 0, 0, 0, 0])),
    pygame.K_h:      (lambda x: x.shearAll([0, 0.1, 0, 0, 0, 0])),
    pygame.K_b:      (lambda x: x.shearAll([0, 0, -0.1, 0, 0, 0])),
    pygame.K_n:      (lambda x: x.shearAll([0, 0, 0.1, 0, 0, 0])),

    pygame.K_m:      (lambda x: x.subdivideAll()), 
    pygame.K_0:      (lambda x: x.centerAll()),
    pygame.K_9:      (lambda x: x.changeAllShapes()),     
    }

class ProjectionViewer:
    """ Displays 3D objects on a Pygame screen """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Wireframe Display')
        pygame.key.set_repeat(1,10)
        self.background = (10,10,50)

        self.wireframes = {}
        self.displayNodes = True
        self.displayEdges = True
        self.nodeColour = (255,255,255)
        self.edgeColour = (200,200,200)
        self.nodeRadius = 4
        
        #custom events
        self.UPDATE_EVENT= pygame.USEREVENT + 1
        pygame.time.set_timer(self.UPDATE_EVENT, 500)
        
        self.ANIMATE_EVENT= pygame.USEREVENT + 2
        pygame.time.set_timer(self.ANIMATE_EVENT, 50)
        
        self.RANDOM_EVENT= pygame.USEREVENT + 3
        pygame.time.set_timer(self.RANDOM_EVENT, 50)
        
        self.animate = False
        self.updateEvent = False
        self.randomEvent = False
        self.showHistory = False
        self.colourShift = False
        self.useWireframeColour = False
        self.useRandomColour = False

        self.movePath_n = 200        
        self.moveIndex = 0
        self.moveCenter = (self.width/2,self.height/2,0)
        
        self.history = []
        self.yoyoCounter = 0
        
    def run(self):
        """ Create a pygame screen until it is closed. """
        self.start_ticks = pygame.time.get_ticks()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in key_to_function:
                        key_to_function[event.key](self)
                    
                elif event.type == self.UPDATE_EVENT:
                    if self.updateEvent:
                        self.update()
                
                elif event.type == self.ANIMATE_EVENT:
                    if self.animate:
                        self.animate_update()                        

                elif event.type == self.RANDOM_EVENT:
                    if self.randomEvent:
                        self.random_update()  
                        
                    
            self.display()
            pygame.display.flip()   
            self.timedAction()

    def addWireframe(self, name, wireframe):
        """ Add a named wireframe object. """ 
        self.wireframes[name] = wireframe  
        self.history.append(name)


    def removeWireframe(self,name):
        try:
            self.wireframes.pop(name)
            self.history.remove(name)
        except:
            print(name, ' not found')

    def timedAction(self):
        self.toRemove = []
        for wireframe in self.wireframes.values():
            t = (datetime.now() - wireframe.initTime_raw).total_seconds() 
            if t > 10:
                self.toRemove.append(wireframe.name)
        if len(self.toRemove) > 0:
            for name in self.toRemove:
                self.removeWireframe(name)
        
                
                

    def getRandomColour(self):
        r = randint(0,255)
        g = randint(0,255)
        b = randint(0,255)
        return (r,g,b)

    def shiftColour(self,colour, count):
        r = colour[0]-(count)
        g = colour[1]-(2*count)
        b = colour[2]-(5*count)
        
        if r <0:
            r=0
        if g<0:
            g=0
        if b<0:
            b=0
        
        a = 1/(count+1)
               
        return (r,g,b,a)

    def display(self):
        """ Draw the wireframes on the screen. """

        self.screen.fill(self.background)

        for wireframe in self.wireframes.values():
            if self.displayEdges:
                for n1, n2 in wireframe.edges:
                    if self.useRandomColour:                    
                        pygame.draw.aaline(self.screen, self.getRandomColour(), wireframe.nodes[n1][:2], wireframe.nodes[n2][:2], 1)
                    else:
                        pygame.draw.aaline(self.screen, self.edgeColour, wireframe.nodes[n1][:2], wireframe.nodes[n2][:2], 1)
                                    
            if self.displayNodes:
                
                if self.useWireframeColour:
                    self.nodeColour = wireframe.getNodeColour()
                
                for node in wireframe.nodes:
                    if self.useRandomColour:
                        pygame.draw.circle(self.screen, self.getRandomColour(), (int(node[0]), int(node[1])), self.nodeRadius, 0)                        
                    else:
                        pygame.draw.circle(self.screen, self.nodeColour, (int(node[0]), int(node[1])), self.nodeRadius, 0)
                
                if self.showHistory:
                    for count, oldnode in enumerate(wireframe.history):
                        for node in oldnode:
                            size = int(self.nodeRadius+count)
                            #size = int(count/self.nodeRadius)
                            if size < 0:
                                size = 0
                            if self.colourShift:
                                colour = self.shiftColour(self.nodeColour,count)
                            else:
                                colour = self.nodeColour
                            pygame.draw.circle(self.screen, colour, (int(node[0]), int(node[1])), size, 0)



    def perspectiveMatrix(self,c):
        """Returns the perspective projection"""
        
        return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1/c, 1]]) 


    def shearMatrix(self,hxy=0, hxz=0, hyz=0, hyx=0, hzx=0, hzy=0):
        """ Return matrix for shear along vector (hxy, hxz, hyz, hyx, hzx, hzy). """
        return np.array([[1,hxy,hxz,0],
                         [hyx,1,hyz,0],
                         [hzx,hzy,1,0],
                         [0,0,0,1]])



    def translationMatrix(self, dx=0, dy=0, dz=0):
        """ Return matrix for translation along vector (dx, dy, dz). """

        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [dx,dy,dz,1]])

    def scaleMatrix(self, sx=0, sy=0, sz=0):
        """ Return matrix for scaling equally along all axes centred on the point (cx,cy,cz). """

        return np.array([[sx, 0,  0,  0],
                         [0,  sy, 0,  0],
                         [0,  0,  sz, 0],
                         [0,  0,  0,  1]])



    def rotateXMatrix(self, radians):
        """ Return matrix for rotating about the x-axis by 'radians' radians """

        c = np.cos(radians)
        s = np.sin(radians)
        return np.array([[1, 0, 0, 0],
                         [0, c,-s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])
    
    def rotateYMatrix(self, radians):
        """ Return matrix for rotating about the y-axis by 'radians' radians """

        c = np.cos(radians)
        s = np.sin(radians)
        return np.array([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]])
    
    def rotateZMatrix(self, radians):
        """ Return matrix for rotating about the z-axis by 'radians' radians """

        c = np.cos(radians)
        s = np.sin(radians)
        return np.array([[c,-s, 0, 0],
                         [s, c, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


    def identityMatrix(self):
        return np.array([[1,0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


    def shearAll(self,vector):
        """ Shear all wireframes along a given axis by d units. """
        
        matrix = self.shearMatrix(vector[0], vector[1], vector[2], vector[0], vector[1], vector[2])
        for wireframe in self.wireframes.values():
            center = wireframe.findCentre()
            wireframe.update('shear',center, matrix)        


    def translateAll(self, vector):
        """ Translate all wireframes along a given axis by d units. """
    
        matrix = self.translationMatrix(vector[0], vector[1], vector[2])
        self.moveCenter = (self.moveCenter[0] + vector[0], self.moveCenter[1] + vector[1], self.moveCenter[2] + vector[2])
        for wireframe in self.wireframes.values():
            wireframe.update('transform', matrix)

    def scaleAll(self, scale):
        """ Scale all wireframes by a given scale, centred on the centre of the screen. """

        center = [self.width/2, self.height/2, 0, 0]
        matrix = self.scaleMatrix(scale, scale, scale)

        for wireframe in self.wireframes.values():
            #print (wireframe.nodes)
            wireframe.update('scale',center, matrix)

    def rotateAll(self, axis, theta):
        """ Rotate all wireframe about their centre, along a given axis by a given angle. """

        rotateFunction = 'rotate' + axis + 'Matrix'

        for wireframe in self.wireframes.values():
            center = wireframe.findCentre()
            matrix = getattr(self, rotateFunction)(theta)
            wireframe.update('rotate',center, matrix)


    def centerAll(self):
        center = [self.width/2, self.height/2, 0, 0]

        for wireframe in self.wireframes.values():
            centerPosition = wireframe.findCentre()
            vector = center - centerPosition
            matrix = self.translationMatrix(vector[0], vector[1], vector[2])
            wireframe.update('transform', matrix)     


    def moveAll(self,x,y,z):
        newPos = [x, y, z, 0]

        for wireframe in self.wireframes.values():
            centerPosition = wireframe.findCentre()
            vector = newPos - centerPosition
            matrix = self.translationMatrix(vector[0], vector[1], vector[2])
            wireframe.update('transform', matrix)          


    def subdivideAll(self):
        for wireframe in self.wireframes.values():
            wireframe.subdivide()

    def toggleEdges(self):
        if self.displayEdges:
            self.displayEdges = False
        else:
            self.displayEdges = True  

    def toggleNodes(self):
        if self.displayNodes:
            self.displayNodes = False
        else:
            self.displayNodes = True  


    def randomColour(self):
        r = randint(0,255)
        g = randint(0,255)
        b = randint(0,255)
        self.nodeColour = (r,g,b)


    def changeAllShapes(self, shape = Tetrahedron(args1=(0,0,0), edgeLength=100) ):
         for wireframe in self.wireframes.values():
            wireframe.changeShape(shape)       


    def animate_update(self):
        self.spin()
        #self.randomMove()        
        #self.moveAroundShape(self.pointsInCircum(100, n=self.movePath_n))
        self.yoyo()
      

    def randomMove(self):
        x = randint(0,self.width)
        y = randint(0,self.height)      
        z = randint(0,100)
        self.moveAll(x,y,z)


    def pointsInCircum(self,r,n=100):
        pi = math.pi
        update = self.moveIndex%n
        return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)][update]


    def moveAroundShape(self,pts):
        x = pts[0] + self.moveCenter[0]
        y = pts[1] + self.moveCenter[1]
        z = 0 + self.moveCenter[2]
        self.moveAll(x,y,z)
        self.moveIndex = self.moveIndex +1
        
        if self.moveIndex > self.movePath_n:
            self.moveIndex = 0

    def zMotion(self,direction='away'):
        if direction == 'away':
            self.scaleAll(0.85) 
        else:
            self.scaleAll(1.1762)

    def yoyo(self):
        if self.yoyoCounter < 20:
            self.zMotion()
        else:
            self.zMotion(direction='towards')   
        self.yoyoCounter +=1
        if self.yoyoCounter >=40:
            self.yoyoCounter = 0

    def spin(self):
        self.translateAll([-10, 0, 0])
        self.rotateAll('Z',  0.1)
        self.translateAll([ 10, 0, 0])
        self.rotateAll('X',  0.1) 
        self.translateAll([0,  10, 0])
        self.rotateAll('Y',  0.1) 
        self.translateAll([0, -10, 0])  

    def update(self):
        for wireframe in self.wireframes.values():
            wireframe.reduceHistory()
        
    def random_update(self):
        self.randomColour()


if __name__ == '__main__':    
    pv = ProjectionViewer(600, 600)
    x,y,z = (300,300,250)
    w,h,d = (50,50,50)
    u,v = (30,30)
    
    wf_models = {'cube' : Cuboid(args1=(x,y,z), args2=(w,h,d)),
               'tetra': Tetrahedron(args1=(0,0,0), edgeLength=100),
               'octa' : Octahedron(),
               'octasphere' : Octasphere(),
               'icosahedron' : Icosahedron (),
               'ball' : Spheroid(args1=(x,y,z), args2=(h, w, d), resolution=10),
               'grid' : HorizontalGrid(args1=(x,y,z), args2=(10,10), args3=(10,20)),
               'fractal': FractalLandscape(origin=(x,y,z), dimensions=(400,400), iterations=5, height=40),
               'teapot' : loadOBJ('teapot.obj'),
               'machingCubes' : marchingCubes(),
               'hull' : convexHull(torusPts=True),
               #'superRes_data' : loadNIKONFile(r"C:\Users\g_dic\OneDrive\Desktop\batchTest\0_trial_1_superes_cropped.txt"),
               #'mri_data' : MRI(),
               'letter' : letter(),
               'param_sphere' : parametric_surface(u, v, sphere),
               'param_torus' : parametric_surface(u, v, torus),               
               }
    
    wf_name = 'octasphere'
    pv.addWireframe(wf_name, wf_models[wf_name])
    pv.centerAll()
     
    
    #imageTest = loadImage(r"C:\Users\g_dic\Pictures\Camera Roll\WIN_20200722_20_16_37_Pro.jpg")
    #imageTest = loadImageLetter(r"C:\Users\g_dic\Pictures\Camera Roll\A.jpg", sigma=3)  
    #pv.addWireframe('imageTest', imageTest) 

    
    pv.run()
    
    
    
    
    
    