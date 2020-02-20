import bpy
import os
import numpy as np
import sys
import argparse


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    # >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ("--"). The approach is that all
    arguments before "--" go to Blender, arguments after go to the script.
    The following calls work fine:
    # >>> blender --python my_script.py -- -a 1 -b 2
    # >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the "--" element (if present, otherwise returns
        an empty list).
        """
        idx = sys.argv.index("--")
        return sys.argv[idx + 1:]  # the list after "--"

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def main():

    """
    This function takes in a  mesh and projects it on a sphere to createa more
    uniform mesh. It saves this mesh both as an .stl and .blend mesh and saves
    the new mesh vertices and faces as numpy arrays (.npy).

    This is run in bash, and inputs are not passed in the parentheses as function
    parameters but rather as inputs with flags.

    <Blender_filepath> -b -P <This_file_filepath> -- (cont next line)
    -i <path_input_mesh> -o <path_output> -d  <mesh_density>

    Parameters
    ----------
    input_filepath: str
        Filepath to input mesh (.stl)

    output_filepath: str
        Filepath base to store results, extended as follows to store:
        Mesh: `.stl`
        Mesh: `.blend`
        Vertices: `_vertices.npy`
        Faces: `_faces.npy`

    mesh_density: int
        Integer 1-10 setting mesh density (higher int gives higher density)
    """

    p = ArgumentParserForBlender(
        prog="uniform_trimesh",
        description="Make mesh more uniform"
    )
    p.add_argument(
        "-i",
        "--input_filepath",
        #  type=str,
        help="filepath to input mesh",
    )
    p.add_argument(
        "-o",
        "--output_filepath",
        #  type=str,
        help="filepath to store colored mesh output file",
    )
    p.add_argument(
        "-d",
        "--mesh_density",
        #  type=int,
        help="int 1-10 giving mesh density",
    )

    p = p.parse_args()
    path_input_mesh = p.input_filepath
    path_output = p.output_filepath
    mesh_density = int(p.mesh_density)

    #  Object name must be file name with underscores changed to spaces
    #  and first letter of each word capitalized
    object_name = path_input_mesh.replace("_", " ").title()

    # delete default blender cube
    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.object.delete()

    # import input mesh STL and center it 
    # (centering should probably be done earlier in pipeline, but doing it here for now)

    bpy.ops.import_mesh.stl(filepath=path_input_mesh)
    input_mesh = bpy.data.objects[object_name]
    input_mesh.location.x = -100.
    input_mesh.location.y = -100.
    input_mesh.location.z = -100.
    input_mesh.name = "Input Mesh"

    # create a sphere with icosahedral triangle mesh
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=mesh_density, 
        radius=100.0, 
        align="WORLD", 
        location=(0.0, 0.0, 0.0), 
        rotation=(0.0, 0.0, 0.0)
    )

    # shrinkwrap sphere onto input mesh
    output_mesh = bpy.data.objects["Icosphere"]
    output_mesh.name = "Output Mesh"
    modifier = output_mesh.modifiers.new("shrinkwrap", "SHRINKWRAP")
    modifier.target = input_mesh
    output_mesh.select_set(True)
    bpy.ops.object.modifier_apply(apply_as="DATA", modifier="shrinkwrap")

    # export STL file
    input_mesh.select_set(False)
    output_mesh.select_set(True)
    mesh_output_path = "{}.stl".format(path_output)
    if os.path.exists(mesh_output_path):
        os.remove(mesh_output_path)
    bpy.ops.export_mesh.stl(
        filepath=mesh_output_path, 
        check_existing=False, 
        filter_glob="*.stl", 
        use_selection=True, 
        global_scale=1.0, 
        use_scene_unit=False, 
        ascii=False, 
        use_mesh_modifiers=True, 
        batch_mode="OFF", 
        axis_forward="Y", 
        axis_up="Z"
    )

    # export vertices
    vertices_output_path = "{}_vertices.npy".format(path_output)
    vertices = []
    for vertex in bpy.data.meshes["Icosphere"].vertices:
        vertices.append([vertex.co[0], vertex.co[1], vertex.co[2]])
    np.save(vertices_output_path, vertices)

    # export faces
    faces_output_path = "{}_faces.npy".format(path_output)
    faces = []
    for face in bpy.data.meshes["Icosphere"].polygons:
        faces.append([face.vertices[0], face.vertices[1], face.vertices[2]])
    np.save(faces_output_path, faces)

    # save blender file for viewing overlaid input and output meshes
    blender_output_path = "{}.blend".format(path_output)
    if os.path.exists(blender_output_path):
        os.remove(blender_output_path)
    bpy.ops.wm.save_as_mainfile(filepath=blender_output_path)


if __name__ == "__main__":
    main()
