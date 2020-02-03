import numpy as np
import math
import itertools
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class Mesh():

    def __init__(self, verts, faces, name):
        """Creates a mesh object in any # of spatial dimensions
        :param verts: List - each vertex is a list of [x,y,z]
        :param faces: List - each face is a list of vertices
        :param name: name for this mesh, to identify/save/label
        """
        self.name = name
        self.verts = verts
        self.faces = faces
        self.npts = int(verts.shape[0])
        self.ndims = int(verts[0].shape[0])


# set shorthand param for icosphere generation
ico = (1 + np.sqrt(5)) / 2

# test model definitions: mass positions or mesh vertices
model_verts = {

    # 1D 2 mass line
    '1D_2masses': np.array([[0.], [1.]]),

    # 1D 3 mass line
    '1D_3masses': np.array([[0.], [1.], [2., ]]),

    # 3D 2 mass line
    '3D_2masses': np.array([[0., 0., 0.], [1., 0., 0.]]),

    # 2D square
    '2D_square': np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]]),

    # 2D rectangle
    '2D_rectangle': np.array([[0., 0.], [0., 1.], [0., 2.], 
                             [1., 2.], [1., 1.], [1., 0.]]),

    # 3D cube
    '3D_cube': np.array([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.],
                        [0., 0., 1.], [0., 1., 1.], [1., 1., 1.], [1., 0., 1.]]),

    # 2D triangle
    '2D_triangle': np.array([[1., 0.], [0.5, np.sqrt(3) / 2.], [0., 0.], ]),

    # 3D tetrahedron
    '3D_tetrahedron': np.array([[1., 0., 0.], [0.5, np.sqrt(3) / 2., 0.],
                               [0., 0., 0.], [np.sqrt(3) / 2., np.sqrt(3) / 2., 1.]]),

    # 3D icosahedron
    '3D_icosphere': np.array([[-1, ico, 0], [1, ico, 0], [-1, -ico, 0], [1, -ico, 0],
                              [0, -1, ico], [0, 1, ico], [0, -1, -ico], [0, 1, -ico], 
                              [ico, 0, -1], [ico, 0, 1], [-ico, 0, -1], [-ico, 0, 1]])

}

# test model definitions: mesh connectivities
model_faces = {
    '1D_2masses': [[0, 1]],
    '1D_3masses': [[0, 1],[1, 2]],
    '3D_2masses': [[0, 1]],
    '2D_square': [[0, 1], [1, 2], [2, 3], [3, 0]],
    '2D_rectangle': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
    '3D_cube': [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 0],
                [0, 4], [1, 5], [2, 6], [3, 7]],
    '2D_triangle': [[0, 1], [1, 2], [2, 0]],
    '3D_tetrahedron': [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]],
    '3D_icosphere': [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                     [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                     [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                     [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
}


def mesh_from_models(mesh_label):
    """Creates a mesh object from preset models, by selecting that model's vertices and 
    faces from the respective dictionaries."""
    return Mesh(model_verts[mesh_label], model_faces[mesh_label], mesh_label)


def get_param(param_str, mesh_label):
    """
    Helper function to allow mesh generating functions set model parameters from labels
    :param mesh_label: label for mesh, used to determine parameter values
    :param param_str: string to search label for, to get paramter value
    """

    defaults = {'R': 5, 'N': 5, 'S': 1}

    if param_str in mesh_label:
        start_ind = mesh_label.find(param_str) + 1
        end_ind = mesh_label[start_ind:].find('_')
        if end_ind == -1:
            val = int(mesh_label[start_ind:])
        else:
            val = int(mesh_label[start_ind:end_ind])
        return val
    else:
        return defaults[param_str]


def polygon_mesh(mesh_label):
    """Creates a mesh of an N-sided polygon with radius r. It can be fully connected or not.
    :param mesh_label: label indicating radius r, sides N, and if fully connected (x)
    :return: Mesh object describing N-sided polygon with radius r.
    """

    R = get_param('R', mesh_label)
    N = get_param('N', mesh_label)
    tmp = 2 * np.pi / N
    verts = [R * np.array((math.cos(tmp * n), math.sin(tmp * n))) for n in range(0, N)]

    if 'x' in mesh_label:
        faces = fully_connect_mesh(verts)
    else:
        faces = []
        for i in range(N - 1):
            faces.append([i, i + 1])
        faces.append([N - 1, 0])

    return Mesh(np.array(verts), np.array(faces), mesh_label)


def volume_trimesh(mesh_label):
    """Creates a 3D surface mesh that approaches a sphere as vertices are added.
    :param mesh_label: label indicating radius r, meshing step size S, connection (x)
    :return: Mesh object describing 3D surface with approximate radius r.
    """

    R = get_param('R', mesh_label)
    S = get_param('S', mesh_label)

    # Create spherical mask
    size = 2 * R + 3
    center = np.array([(size - 1) / 2, (size - 1) / 2, (size - 1) / 2])
    mask = np.zeros((size, size, size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if np.linalg.norm(np.array([i, j, k]) - center) <= R:
                    mask[i, j, k] = 1

    # mesh the mask into verts and faces
    verts, faces, n, v = measure.marching_cubes_lewiner(mask, step_size=S)

    if 'x' in mesh_label:
        faces = fully_connect_mesh(verts)
    return Mesh(verts, faces, mesh_label)


def fully_connect_mesh(verts):
    """Connects all vertices in input mesh to all other vertices.
    :param verts: mesh vertices
    :return: faces of mesh, connecting all vertices to all others
    """
    return list(itertools.combinations(range(len(verts)), 2))


def draw_mesh(mesh):
    """Draws mesh if mesh is in 1D or 2D. If 3D, just plots vertices.
    :param mesh: mesh object (see mesh.py)
    """
    
    fig = plt.figure()
    
    # if mesh is 1D, set y to zero and plot vertices and faces along x axis
    if mesh.ndims == 1:
        for pair in mesh.faces:
            x = [mesh.verts[pair[0]][0], mesh.verts[pair[1]][0]]
            y = np.zeros(len(x))
            plt.plot(x, y, marker='o', color='k')

    # if mesh is 2D, plot 2D mesh vertices and faces
    if mesh.ndims == 2:
        for pair in mesh.faces:
            x = [mesh.verts[pair[0]][0], mesh.verts[pair[1]][0]]
            y = [mesh.verts[pair[0]][1], mesh.verts[pair[1]][1]]
            plt.plot(x, y, marker='o', color='k')
            axis = plt.gca()
            axis.set_aspect('equal')

    # if mesh is 3D, just plot vertices in 3D projection
    if mesh.ndims == 3:
        ax = fig.add_subplot(111, projection='3d')
        for vert in mesh.verts:
            ax.scatter(vert[0], vert[1], vert[2], color='k')

    return fig
