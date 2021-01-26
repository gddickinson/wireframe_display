import pygame_MatrixWireframe as wf
import pygame, sys
import numpy as np


# matrix transforms
key_to_function = {
 pygame.K_LEFT: (lambda x: x.translateAll([-10, 0, 0])),
 pygame.K_RIGHT:(lambda x: x.translateAll([ 10, 0, 0])),
 pygame.K_DOWN: (lambda x: x.translateAll([0,  10, 0])),
 pygame.K_UP:   (lambda x: x.translateAll([0, -10, 0])),

class ProjectionViewer:
    """ Displays 3D objects on a Pygame screen """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Wireframe Display')
        self.background = (10,10,50)
        
        self.wireframes = {}
        self.displayNodes = True
        self.displayEdges = True
        self.nodeColour = (255,255,255)
        self.edgeColour = (200,200,200)
        self.nodeRadius = 4
        
    def run(self):
        """ Create a pygame screen until it is closed. """

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
                    
            self.display()
            pygame.display.flip()        

    def addWireframe(self, name, wireframe):
        """ Add a named wireframe object. """
        self.wireframes[name] = wireframe        


    def display(self):
        """ Draw the wireframes on the screen. """

        self.screen.fill(self.background)

        for wireframe in self.wireframes.values():
            if self.displayEdges:
                for n1, n2 in wireframe.edges:
                    pygame.draw.aaline(self.screen, self.edgeColour, wireframe.nodes[n1][:2], wireframe.nodes[n2][:2], 1)
                
            if self.displayNodes:
                for node in wireframe.nodes:
                    pygame.draw.circle(self.screen, self.nodeColour, (int(node[0]), int(node[1])), self.nodeRadius, 0)

    def translateAll(self, vector):
        """ Translate all wireframes along a given axis by d units. """
    
        matrix = wf.translationMatrix(*vector)
        for wireframe in self.wireframes.itervalues():
            wireframe.transform(matrix)



if __name__ == '__main__':
    cube = wf.Wireframe()
    cube.addNodes(np.array(cube_nodes))
    cube.addEdges([(n,n+4) for n in range(0,4)]+[(n,n+1) for n in range(0,8,2)]+[(n,n+2) for n in (0,1,4,5)])
    
    pv = ProjectionViewer(400, 300)
    pv.addWireframe('cube', cube)
    pv.run()
    
    
    
    
    
    