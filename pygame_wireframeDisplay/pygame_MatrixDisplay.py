"""Pygame window for the matrix wireframe viewer: keyboard control, timed animation events and drawing.

The transform maths and scene operations live in wireframe_scene.WireframeScene (Pygame-free, shared with the web
viewer); this module adds the window, the event loop and display(). Run: python pygame_MatrixDisplay.py
"""
import pygame_MatrixWireframe as wf
import pygame, sys
import numpy as np
from pygame_basicShapes import *
from pygame_parametricShapes import *
import time
from random import randint
import math
from datetime import datetime, timedelta
from wireframe_scene import KEY_BINDINGS, WireframeScene

# matrix transforms: key -> scene operation, from the shared table in wireframe_scene.py
key_to_function = {getattr(pygame, 'K_' + key): (lambda x, m=method, a=args: getattr(x, m)(*a))
                   for key, method, args, _ in KEY_BINDINGS}

class ProjectionViewer(WireframeScene):
    """ Displays 3D objects on a Pygame screen """

    def __init__(self, width, height):
        WireframeScene.__init__(self, width, height)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Wireframe Display')
        pygame.key.set_repeat(1,10)
        self.background = (10,10,50)

        #custom events
        self.UPDATE_EVENT= pygame.USEREVENT + 1
        pygame.time.set_timer(self.UPDATE_EVENT, 500)

        self.ANIMATE_EVENT= pygame.USEREVENT + 2
        pygame.time.set_timer(self.ANIMATE_EVENT, 50)

        self.RANDOM_EVENT= pygame.USEREVENT + 3
        pygame.time.set_timer(self.RANDOM_EVENT, 50)


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
            self.timedAction(datetime.now())  # timed demo (off unless started, see --demo)


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





if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Matrix wireframe viewer')
    ap.add_argument('--demo', type=float, nargs='?', const=10, metavar='SECONDS',
                    help='timed demo: show each model for SECONDS (default 10), then the next')
    demo_seconds = ap.parse_args().demo

    pv = ProjectionViewer(600, 600)
    x,y,z = (300,300,250)
    w,h,d = (50,50,50)
    u,v = (30,30)
    
    # Build only the model that is shown: teapot.obj is not in the repository, and building every
    # model up front made the script crash at start-up when it was missing.
    wf_models = {'cube' : lambda: Cuboid(args1=(x,y,z), args2=(w,h,d)),
               'tetra': lambda: Tetrahedron(args1=(0,0,0), edgeLength=100),
               'octa' : lambda: Octahedron(),
               'octasphere' : lambda: Octasphere(),
               'icosahedron' : lambda: Icosahedron (),
               'ball' : lambda: Spheroid(args1=(x,y,z), args2=(h, w, d), resolution=10),
               'grid' : lambda: HorizontalGrid(args1=(x,y,z), args2=(10,10), args3=(10,20)),
               'fractal': lambda: FractalLandscape(origin=(x,y,z), dimensions=(400,400), iterations=5, height=40),
               'teapot' : lambda: loadOBJ('teapot.obj'),
               'machingCubes' : lambda: marchingCubes(),
               'hull' : lambda: convexHull(torusPts=True),
               #'superRes_data' : loadNIKONFile(r"C:\Users\g_dic\OneDrive\Desktop\batchTest\0_trial_1_superes_cropped.txt"),
               #'mri_data' : MRI(),
               'letter' : lambda: letter(),
               'param_sphere' : lambda: parametric_surface(u, v, sphere),
               'param_torus' : lambda: parametric_surface(u, v, torus),               
               }
    
    wf_name = 'octasphere'
    if demo_seconds:
        pv.startTimedDemo([(k, f) for k, f in wf_models.items() if k != 'teapot'], lifetime=demo_seconds)
    else:
        pv.addWireframe(wf_name, wf_models[wf_name]())
        pv.centerAll()
     
    
    #imageTest = loadImage(r"C:\Users\g_dic\Pictures\Camera Roll\WIN_20200722_20_16_37_Pro.jpg")
    #imageTest = loadImageLetter(r"C:\Users\g_dic\Pictures\Camera Roll\A.jpg", sigma=3)  
    #pv.addWireframe('imageTest', imageTest) 

    
    pv.run()
    
    
    
    
    
    