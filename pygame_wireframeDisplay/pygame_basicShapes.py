import numpy as np
import pygame_MatrixWireframe as wf
from scipy.spatial import Delaunay
from math import *
import pyrr
quaternion = pyrr.quaternion

def Cuboid(args1=(0,0,0), args2=(0,0,0)):
    """ Return a wireframe cuboid starting at (x,y,z)
        with width, w, height, h, and depth, d. """
        
    x = args1[0]
    y = args1[1]
    z = args1[2]
    
    w = args2[0]
    h = args2[1]
    d = args2[2]    
    
    cuboid = wf.Wireframe()
    cuboid.addNodes(np.array([[nx,ny,nz] for nx in (x,x+w) for ny in (y,y+h) for nz in (z,z+d)]))
    cuboid.addFaces([(0,1,3,2), (7,5,4,6), (4,5,1,0), (2,3,7,6), (0,2,6,4), (5,7,3,1)])
    
    cuboid.name = 'cube'
    
    return cuboid

def Tetrahedron(args1=(0,0,0), edgeLength=1):

    x = args1[0]
    y = args1[1]
    z = args1[2]
    
    
    A=np.array([edgeLength+x, edgeLength+y, edgeLength+z])
    B=np.array([edgeLength+x, -edgeLength+y, -edgeLength+z])
    C=np.array([-edgeLength+x, edgeLength+y, -edgeLength+z])
    D=np.array([-edgeLength+x, -edgeLength+y, edgeLength+z])
    
    tetrahedron = wf.Wireframe()
    tetrahedron.addNodes(np.array([A,B,C,D]))
    tetrahedron.addFaces([(0,1,2),(2,3,0),(1,3,2),(3,1,0)])    

    tetrahedron.name = 'tetra'

    return tetrahedron

def Octahedron():
    """Construct an eight-sided polyhedron"""
    f =  np.sqrt(2.0) / 2.0
    verts = np.float32([ ( 0, -1,  0), (-f,  0,  f), ( f,  0,  f), ( f,  0, -f), (-f,  0, -f), ( 0,  1,  0) ])
    triangles = np.int32([ (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 1, 4), (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 4, 1) ])
    
    octahedron = wf.Wireframe()
    octahedron.addNodes(verts)
    octahedron.addFaces(triangles)      
    
    return octahedron
    
def Spheroid(args1=(0,0,0), args2=(0, 0, 0), resolution=10):
    """ Returns a wireframe spheroid centred on (x,y,z)
        with a radii of (rx,ry,rz) in the respective axes. """

    x = args1[0]
    y = args1[1]
    z = args1[2]
    
    rx = args2[0]
    ry = args2[1]
    rz = args2[2] 

    
    spheroid   = wf.Wireframe()
    latitudes  = [n*np.pi/resolution for n in range(1,resolution)]
    longitudes = [n*2*np.pi/resolution for n in range(resolution)]

    # Add nodes except for poles
    spheroid.addNodes([(x + rx*np.sin(n)*np.sin(m), y - ry*np.cos(m), z - rz*np.cos(n)*np.sin(m)) for m in latitudes for n in longitudes])

    # Add square faces to whole spheroid but poles
    num_nodes = resolution*(resolution-1)
    spheroid.addFaces([(m+n, (m+resolution)%num_nodes+n, (m+resolution)%resolution**2+(n+1)%resolution, m+(n+1)%resolution) for n in range(resolution) for m in range(0,num_nodes-resolution,resolution)])

    # Add poles and triangular faces around poles
    spheroid.addNodes([(x, y+ry, z),(x, y-ry, z)])
    spheroid.addFaces([(n, (n+1)%resolution, num_nodes+1) for n in range(resolution)])
    start_node = num_nodes-resolution
    spheroid.addFaces([(num_nodes, start_node+(n+1)%resolution, start_node+n) for n in range(resolution)])

    spheroid.name = 'spheroid'

    return spheroid
    
def HorizontalGrid(args1=(0,0,0), args2=(0,0), args3=(0,0)):
    """ Returns a nx by nz wireframe grid that starts at (x,y,z) with width dx.nx and depth dz.nz. """

    x = args1[0]
    y = args1[1]
    z = args1[2]
    
    dx = args2[0]
    dz = args2[1]

    nx = args3[0]
    nz = args3[1]
    
    grid = wf.Wireframe()
    grid.addNodes([[x+n1*dx, y, z+n2*dz] for n1 in range(nx+1) for n2 in range(nz+1)])
    grid.addEdges([(n1*(nz+1)+n2,n1*(nz+1)+n2+1) for n1 in range(nx+1) for n2 in range(nz)])
    grid.addEdges([(n1*(nz+1)+n2,(n1+1)*(nz+1)+n2) for n1 in range(nx) for n2 in range(nz+1)])
    
    grid.name = 'grid'
    
    return grid
    
def FractalLandscape(origin=(0,0,0), dimensions=(400,400), iterations=4, height=40):
    import random
    
    def midpoint(nodes):
        m = 1.0/ len(nodes)
        x = m * sum(n[0] for n in nodes) 
        y = m * sum(n[1] for n in nodes) 
        z = m * sum(n[2] for n in nodes) 
        return [x,y,z]
    
    (x,y,z) = origin
    (dx,dz) = dimensions
    nodes = [[x, y, z], [x+dx, y, z], [x+dx, y, z+dz], [x, y, z+dz]]
    edges = [(0,1), (1,2), (2,3), (3,0)]
    size = 2

    for i in range(iterations):
        # Add nodes midway between each edge
        for (n1, n2) in edges:
            nodes.append(midpoint([nodes[n1], nodes[n2]]))

        # Add nodes to the centre of each square
        squares = [(x+y*size, x+y*size+1, x+(y+1)*size+1, x+(y+1)*size) for y in range(size-1) for x in range(size-1)]
        for (n1,n2,n3,n4) in squares:
            nodes.append(midpoint([nodes[n1], nodes[n2], nodes[n3], nodes[n4]]))
        
        # Sort in order of grid
        nodes.sort(key=lambda node: (node[2],node[0]))
        
        size = size*2-1
        # Horizontal edge
        edges = [(x+y*size, x+y*size+1) for y in range(size) for x in range(size-1)]
        # Vertical edges
        edges.extend([(x+y*size, x+(y+1)*size) for x in range(size) for y in range(size-1)])
        
        # Shift node heights
        scale = height/2**(i*0.8)
        for node in nodes:
            node[1] += (random.random()-0.5)*scale
    
    grid = wf.Wireframe(nodes)
    grid.addEdges(edges)
    
    return grid


def loadOBJ(filename):
    
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
    
    vertices, triangles = read_obj(filename)
    
    model = wf.Wireframe()
    model.addNodes(vertices)
    model.addFaces(triangles)
    
    return model
    
def marchingCubes():
    """" Test of marching cubes function"""
    from skimage import measure
    from skimage.draw import ellipsoid
    
    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    ellip_double = np.concatenate((ellip_base[:-1, ...],
                                   ellip_base[2:, ...]), axis=0)
    
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes_lewiner(ellip_double, 0)
    

    model = wf.Wireframe()
    model.addNodes(verts)
    model.addFaces(faces)

    return model       

def MRI():
    """" Load MRI data""" 
    from nilearn import image

    filename = r"C:\Users\g_dic\OneDrive\Desktop\brain\test4d.nii.gz"
    
    rsn = image.load_img(filename)    
    first_rsn = image.index_img(rsn, 0)    
        
    from skimage import measure
    
    # Use marching cubes to obtain the surface mesh 
    
    data = first_rsn.dataobj
    data[data < 60] = 0
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(data, level=5,step_size=3)

    model = wf.Wireframe()
    model.addNodes(verts)
    model.addFaces(faces)
    
    
    from scipy.spatial import ConvexHull 
    hull = ConvexHull(verts)
    hullShape = wf.Wireframe()
    hullShape.addNodes(verts)
    hullShape.addFaces(hull.simplices)  
    
    return hullShape


def convexHull(pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], ]),torusPts=False, clusteredPts=False, uniformPts=False, edgePts=False):
    """" Test of scipy.spatial convexHull function"""    
    from scipy.spatial import ConvexHull  
    
    if torusPts:
        def rand_torus(ro=1.0, ri=0.1, npoints=500):
            """Generate points within a torus wth major radius `ro` and minor radius
            `ri` via rejection sampling"""
            out = np.empty((npoints, 3), dtype=np.float64)
            i = 0
            while i < npoints:
                # generate point within box
                x = np.random.uniform(-1., 1., size=3)
                x[0:2] *= ro + ri
                x[2] *= ri
        
                r = np.sqrt(x[0]**2 + x[1]**2) - ro
                if (r**2 + x[2]**2 < ri**2):
                    out[i, :] = x
                    i += 1
        
            return out
        
        pts = rand_torus()
        print(pts.shape)
    
    if clusteredPts:
        def clusteredPoints(n=100):
            radius = np.random.uniform(0.0,1.0, (n,1)) 
            theta = np.random.uniform(0.,1.,(n,1))* np.pi
            phi = np.arccos(1-2*np.random.uniform(0.0,1.,(n,1)))
            x = radius * np.sin( theta ) * np.cos( phi )
            y = radius * np.sin( theta ) * np.sin( phi )
            z = radius * np.cos( theta )
            array = np.array([x,y,z]).reshape(n,3)
            return array
        
        pts = clusteredPoints(n=100)        


    if uniformPts:
        from scipy.special import gammainc

        def sample(center = np.array([0,0,0]),radius = 1,n_per_sphere=100):
            r = radius
            ndim = center.size
            x = np.random.normal(size=(n_per_sphere, ndim))
            ssq = np.sum(x**2,axis=1)
            fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
            frtiled = np.tile(fr.reshape(n_per_sphere,1),(1,ndim))
            p = center + np.multiply(x,frtiled)
            return p
               
        pts = sample()

    if edgePts:
        size = 1000
        n = 3 # or any positive integer
        x = np.random.normal(size=(size, n)) 
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
        pts = x

    
    hull = ConvexHull(pts)
    
    hullShape = wf.Wireframe()
    hullShape.addNodes(pts)
    hullShape.addFaces(hull.simplices)   
    
    return hullShape
    

def loadNIKONFile(filename):
    
    def evaluate(i):
    	try:
    		return eval(i)
    	except:
    		return i
    
    def importFile(filename, delimiter="\t", columns=[], evaluateLines=True):
    	'''read info from a file, into a list of columns (specified by args) or dictionaries (specified by kargs)'''
    	data = []
    	if evaluateLines: 
    		lines = [[evaluate(i) for i in line.split(delimiter)] for line in open(filename, 'r')]
    	else:
    		lines = [[i for i in line.split(delimiter)] for line in open(filename, 'r')]        
    
    	if len(columns) == 0: # no columns given, return data as it is read from file
    		return lines
    	else:
    		names = lines[0]    # read data to dictionary given columns
    		lines = lines[1:]
    		if all([type(i) == str for i in columns]):
    			data = {}
    			for n in columns:
    				data[n] = [lines[i][names.index(n)] for i in range(len(lines))]
    		elif all([type(i) == int for i in columns]):
    			data = {}
    			for i in columns:
    				n = names[i]
    				data[n] = [lines[j][i] for j in range(len(lines))]
    		return data

    data = importFile(filename,evaluateLines=False)
        
    try:        
        for i in range(len(data[0])):
            if '\n' in data[0][i]:
                data[0][i] = data[0][i].split('\n')[0]                
    	
    except:
        print('Data load failed')    

    #colNames = list(data[0])
    
    data = {d[0]: d[1:] for d in np.transpose(data)}
   
    for k in data:
        if k != 'Channel Name':
            data[k] = data[k].astype(float)
    print('Gathering channels...')
    ignore = {"Z Rejected"}                      
    names = set(data['Channel Name'].astype(str)) - ignore
    print('Channels Found: {}'.format(names))
     
    unitPerPixel = 166
    # data is loaded in nanometers, divided by # according to units
    units = {'Pixels': unitPerPixel, 'Nanometers': 1}
    unit = 'Nanometers'  
     
    data['Xc'] /= units[unit]
    data['Yc'] /= units[unit]
    data['Zc'] /= units[unit]
    
    n = len(data['Xc'])
    
    array = np.array([data['Xc'],data['Yc'],data['Zc']]).reshape(n,3)
    
    shape = wf.Wireframe()
    shape.addNodes(array)
    
    return shape


def letter():
    
    nodesDict = {}
    facesDict = {}

    A=np.array([0,0,0])
    B=np.array([4,0,0])
    C=np.array([0,0,2])
    D=np.array([4,0,2])
    E=np.array([14, 0, 0])
    F=np.array([18, 0, 0])
    G=np.array([14, 0, 2])
    H=np.array([18, 0, 2])
    I=np.array([5, 6, 0])
    J=np.array([13, 6, 0])
    K=np.array([5, 6, 2])
    L=np.array([13, 6, 2])
    M=np.array([7, 13, 0])
    N=np.array([11, 13, 0])
    O=np.array([7, 13, 2])
    P=np.array([11, 13, 2])
    Q=np.array([9, 17, 0])
    R=np.array([9, 17, 2])
    S=np.array([7, 24, 0])
    T=np.array([11, 24, 0])
    U=np.array([7, 24, 2])
    V=np.array([11, 24, 2])
    
    nodesDict['A'] = np.array([A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V])
   
    facesDict['A'] = np.array([(0,1,3,2),(4,5,7,6),(8,9,11,10),(12,13,15,14),(18,19,21,20),
                     (0,2,20,18),(5,7,21,19),(1,3,10,8),(4,6,11,9),(12,14,17,16),(13,15,17,16)])   
       
    letter = wf.Wireframe()
    letter.addNodes(nodesDict['A'])
    letter.addFaces(facesDict['A'])  
    
    return letter

def loadImage(fileName, sigma=3):
    from matplotlib import pyplot as plt
    from PIL import Image
    from skimage import feature
    from scipy.ndimage import gaussian_filter
    image = Image.open(fileName).convert('L')
    
    data = np.asarray(image)
    #data = data[0:200,0:200]
    #data = gaussian_filter(data, sigma=5)
    
    edges = feature.canny(data, sigma=sigma) 
    
    indices = np.where(edges != [0])
    
    x,y = indices[0],indices[1]    

    #plt.scatter(x,y)  
    
    z= np.ones_like(y)
    n = len(y)
    
    pts = np.dstack([x,y,z])[0]

    
    image2D = wf.Wireframe()
    image2D.addNodes(pts)
    #imagePoint.addFaces() 
    return image2D


def loadImageLetter(fileName, sigma=3):
    from matplotlib import pyplot as plt
    from PIL import Image
    from skimage import feature
    from scipy.ndimage import gaussian_filter
    from scipy.signal import convolve2d
    from skimage import measure
    import alphashape
    image = Image.open(fileName).convert('L')
    
    data = np.asarray(image)
    #data = data[0:200,0:200]
    data = gaussian_filter(data, sigma=5)
    
    edges = feature.canny(data, sigma=sigma)
    
    indices = np.where(edges != [0])
   
    x,y = indices[0],indices[1]    

    #alpha shape
    pts_XY = np.dstack([x,y])[0]
    alpha_shape = alphashape.alphashape(pts_XY, 0.1)   #to solve for optimal leave second paramter empty 
    x,y = alpha_shape.boundary.xy
    
    #plt.scatter(x,y)  
    
    z= np.ones_like(y) * 30   
    z2 = np.zeros_like(y)
    
    x1 = np.concatenate([x,x])
    y1 = np.concatenate([y,y])    
    z1 = np.concatenate([z,z2])      
    
    pts = np.dstack([x1,y1,z1])[0]
    
    image2D = wf.Wireframe()
    image2D.addNodes(pts)
    #image2D.addEdges() 
    return image2D



def parametric_surface(slices, stacks, func):
    verts = []
    for i in range(slices + 1):
        theta = i * pi / slices
        for j in range(stacks):
            phi = j * 2.0 * pi / stacks
            p = func(theta, phi)
            verts.append(p)
    verts = np.float32(verts)

    faces = []
    v = 0
    for i in range(slices):
        for j in range(stacks):
            next = (j + 1) % stacks
            faces.append((v + j, v + j + stacks, v + next + stacks, v + next))
        v = v + stacks
    faces = np.int32(faces)


    paramSurf = wf.Wireframe()
    paramSurf.addNodes(verts)
    paramSurf.addFaces(faces) 
    return paramSurf


def get_octasphere(ndivisions: int, radius: float, width=0, height=0, depth=0):
    """Generates a triangle mesh for a sphere, rounded cube, or capsule.
    The ndivisions argument can be used to control the level of detail
    and should be between 0 and 5, inclusive.
    To create a sphere, simply omit the width/height/depth arguments.
    To create a capsule, set one of width/height/depth to a value
    greater than twice the radius. To create a cuboid, set two or more
    of these to a value greater than twice the radius.
    Returns a two-tuple: a numpy array of 3D vertex positions,
    and a numpy array of integer 3-tuples for triangle indices.
    """
    r2 = 2 * radius
    width = max(width, r2)
    height = max(height, r2)
    depth = max(depth, r2)
    n = 2**ndivisions + 1
    num_verts = n * (n + 1) // 2
    verts = np.empty((num_verts, 3))
    j = 0
    for i in range(n):
        theta = pi * 0.5 * i / (n - 1)
        point_a = [0, sin(theta), cos(theta)]
        point_b = [cos(theta), sin(theta), 0]
        num_segments = n - 1 - i
        j = compute_geodesic(verts, j, point_a, point_b, num_segments)
    assert len(verts) == num_verts
    verts = verts * radius

    num_faces = (n - 2) * (n - 1) + n - 1
    faces = np.empty((num_faces, 3), dtype=np.int32)
    f, j0 = 0, 0
    for col_index in range(n-1):
        col_height = n - 1 - col_index
        j1 = j0 + 1
        j2 = j0 + col_height + 1
        j3 = j0 + col_height + 2
        for row in range(col_height - 1):
            faces[f + 0] = [j0 + row, j1 + row, j2 + row]
            faces[f + 1] = [j2 + row, j1 + row, j3 + row]
            f = f + 2
        row = col_height - 1
        faces[f] = [j0 + row, j1 + row, j2 + row]
        f = f + 1
        j0 = j2

    euler_angles = np.float32([
        [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
        [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3],
    ]) * pi * 0.5
    quats = (quaternion.create_from_eulers(e) for e in euler_angles)

    offset, combined_verts, combined_faces = 0, [], []
    for quat in quats:
        rotated_verts = [quaternion.apply_to_vector(quat, v) for v in verts]
        rotated_faces = faces + offset
        combined_verts.append(rotated_verts)
        combined_faces.append(rotated_faces)
        offset = offset + len(verts)

    verts = np.vstack(combined_verts)

    tx = (width - r2) / 2
    ty = (height - r2) / 2
    tz = (depth - r2) / 2
    translation = np.float32([tx, ty, tz])

    if np.any(translation):
        translation = np.float32([
            [+1, +1, +1], [+1, +1, -1], [-1, +1, -1], [-1, +1, +1],
            [+1, -1, +1], [-1, -1, +1], [-1, -1, -1], [+1, -1, -1],
        ]) * translation
        for i in range(0, len(verts), num_verts):
            verts[i:i+num_verts] += translation[i // num_verts]
        connectors = add_connectors(ndivisions, radius, width, height, depth)
        if radius == 0:
            assert len(connectors) // 2 == 6
            combined_faces = connectors
        else:
            combined_faces.append(connectors)

    return verts, np.vstack(combined_faces)

def add_connectors(ndivisions, radius, width, height, depth):
    r2 = 2 * radius
    width = max(width, r2)
    height = max(height, r2)
    depth = max(depth, r2)
    n = 2**ndivisions + 1
    num_verts = n * (n + 1) // 2
    tx = (width - r2) / 2
    ty = (height - r2) / 2
    tz = (depth - r2) / 2

    boundaries = get_boundary_indices(ndivisions)
    assert len(boundaries) == 3
    connectors = []

    def connect(a, b, c, d):
        # if np.allclose(verts[a], verts[b]): return
        # if np.allclose(verts[b], verts[d]): return
        connectors.append([a, b, c])
        connectors.append([c, d, a])

    if radius > 0:
        # Top half
        for patch in range(4):
            if patch % 2 == 0 and tz == 0: continue
            if patch % 2 == 1 and tx == 0: continue
            next_patch = (patch + 1) % 4
            boundary_a = boundaries[1] + num_verts * patch
            boundary_b = boundaries[0] + num_verts * next_patch
            for i in range(n-1):
                a = boundary_a[i]
                b = boundary_b[i]
                c = boundary_a[i+1]
                d = boundary_b[i+1]
                connect(a, b, d, c)
        # Bottom half
        for patch in range(4,8):
            if patch % 2 == 0 and tx == 0: continue
            if patch % 2 == 1 and tz == 0: continue
            next_patch = 4 + (patch + 1) % 4
            boundary_a = boundaries[0] + num_verts * patch
            boundary_b = boundaries[2] + num_verts * next_patch
            for i in range(n-1):
                a = boundary_a[i]
                b = boundary_b[i]
                c = boundary_a[i+1]
                d = boundary_b[i+1]
                connect(d, b, a, c)
        # Connect top patch to bottom patch
        if ty > 0:
            for patch in range(4):
                next_patch = 4 + (4 - patch) % 4
                boundary_a = boundaries[2] + num_verts * patch
                boundary_b = boundaries[1] + num_verts * next_patch
                for i in range(n-1):
                    a = boundary_a[i]
                    b = boundary_b[n-1-i]
                    c = boundary_a[i+1]
                    d = boundary_b[n-1-i-1]
                    connect(a, b, d, c)

    if tx > 0 or ty > 0:
        # Top hole
        a = boundaries[0][-1]
        b = a + num_verts
        c = b + num_verts
        d = c + num_verts
        connect(a, b, c, d)
        # Bottom hole
        a = boundaries[2][0] + num_verts * 4
        b = a + num_verts
        c = b + num_verts
        d = c + num_verts
        connect(a, b, c, d)

    # Side holes
    sides = []
    if ty > 0: sides = [(7,0),(1,2),(3,4),(5,6)]
    for i, j in sides:
        patch_index = i
        patch = patch_index // 2
        next_patch = 4 + (4 - patch) % 4
        boundary_a = boundaries[2] + num_verts * patch
        boundary_b = boundaries[1] + num_verts * next_patch
        if patch_index % 2 == 0:
            a,b = boundary_a[0], boundary_b[n-1]
        else:
            a,b = boundary_a[n-1], boundary_b[0]
        patch_index = j
        patch = patch_index // 2
        next_patch = 4 + (4 - patch) % 4
        boundary_a = boundaries[2] + num_verts * patch
        boundary_b = boundaries[1] + num_verts * next_patch
        if patch_index % 2 == 0:
            c,d = boundary_a[0], boundary_b[n-1]
        else:
            c,d = boundary_a[n-1], boundary_b[0]
        connect(a, b, d, c)

    return connectors


def compute_geodesic(dst, index, point_a, point_b, num_segments):
    """Given two points on a unit sphere, returns a sequence of surface
    points that lie between them along a geodesic curve."""
    angle_between_endpoints = acos(np.dot(point_a, point_b))
    rotation_axis = np.cross(point_a, point_b)
    dst[index] = point_a
    index = index + 1
    if num_segments == 0:
        return index
    dtheta = angle_between_endpoints / num_segments
    for point_index in range(1, num_segments):
        theta = point_index * dtheta
        q = quaternion.create_from_axis_rotation(rotation_axis, theta)
        dst[index] = quaternion.apply_to_vector(q, point_a)
        index = index + 1
    dst[index] = point_b
    return index + 1


def get_boundary_indices(ndivisions):
    "Generates the list of vertex indices for all three patch edges."
    n = 2**ndivisions + 1
    boundaries = np.empty((3, n), np.int32)
    a, b, c, j0 = 0, 0, 0, 0
    for col_index in range(n-1):
        col_height = n - 1 - col_index
        j1 = j0 + 1
        boundaries[0][a] = j0
        a = a + 1
        for row in range(col_height - 1):
            if col_height == n - 1:
                boundaries[2][c] = j0 + row
                c = c + 1
        row = col_height - 1
        if col_height == n - 1:
            boundaries[2][c] = j0 + row
            c = c + 1
            boundaries[2][c] = j1 + row
            c = c + 1
        boundaries[1][b] = j1 + row
        b = b + 1
        j0 = j0 + col_height + 1
    boundaries[0][a] = j0 + row
    boundaries[1][b] = j0 + row
    return boundaries

def Octasphere(w=10,h=10,d=10):
    '''https://github.com/prideout/svg3d'''
    verts, faces =  get_octasphere(3, 20.0, width=w, height=h, depth=d) 

    octasphere = wf.Wireframe()
    octasphere.addNodes(verts)
    octasphere.addFaces(faces)      
    
    return octasphere     

def Icosahedron():
    """Construct a 20-sided polyhedron"""
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 1),
        (11, 7, 6),
        (11, 8, 7),
        (11, 9, 8),
        (11, 10, 9),
        (11, 6, 10),
        (1, 6, 2),
        (2, 7, 3),
        (3, 8, 4),
        (4, 9, 5),
        (5, 10, 1),
        (6, 7, 2),
        (7, 8, 3),
        (8, 9, 4),
        (9, 10, 5),
        (10, 6, 1),
    ]
    verts = [
        (0.000, 0.000, 1.000),
        (0.894, 0.000, 0.447),
        (0.276, 0.851, 0.447),
        (-0.724, 0.526, 0.447),
        (-0.724, -0.526, 0.447),
        (0.276, -0.851, 0.447),
        (0.724, 0.526, -0.447),
        (-0.276, 0.851, -0.447),
        (-0.894, 0.000, -0.447),
        (-0.276, -0.851, -0.447),
        (0.724, -0.526, -0.447),
        (0.000, 0.000, -1.000),
    ]

    icosahedron = wf.Wireframe()
    icosahedron.addNodes(verts)
    icosahedron.addFaces(faces)      
    
    return icosahedron 
    
if __name__ == '__main__':
    #grid = FractalLandscape(origin = (0,400,0), iterations=1)
    #grid.output()
    
    #tetra = Tetrahedron()
    #tetra.output()
    
    #imageTest = loadImage(r"C:\Users\g_dic\OneDrive\Pictures\Camera Roll\WIN_20200722_20_16_37_Pro.jpg")
    #imageTest = loadImageLetter(r"C:\Users\g_dic\OneDrive\Pictures\Camera Roll\A.jpg", sigma=3)      
    #imageTest.output()
    
    hexahedron = Hexahedron()