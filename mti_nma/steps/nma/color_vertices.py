import bpy
import os
import numpy as np
import argparse
import sys


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    # >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    # >>> blender --python my_script.py -- -a 1 -b 2
    # >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        idx = sys.argv.index("--")
        return sys.argv[idx + 1:]  # the list after '--'

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
    This function takes in a mesh (.stl) and the magnitudes of the eigenvectors at
    all vertices in the mesh (.npy) for a given mode and generates a new file of the
    mesh (.blend) colorized with a heatmap of the eigenvector magnitudes.

    This is run in bash, and inputs are not passed in the parentheses as function
    parameters but rather as inputs with flags.

    <Blender_filepath> -b -P <This_file_filepath> -- (cont next line)
    -i <path_input_mesh> -o <path_output> -m  <mode> -v <path_vmags>

    Parameters
    ----------
    input_filepath: str
        Filepath to input mesh (.stl)

    output_filepath: str
        Filepath to store colorized output mesh (.blend)

    vmag_filepath: str
        Filepath to amplitudes of mode eigenvectors (.npy)

    mode_index: int
        Index of mode to display/save mesh colorizations for
    """

    p = ArgumentParserForBlender(
        prog='color_vertices',
        description="Color vertices by magnitude"
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
        "-m",
        "--mode_index",
        #  type=int,
        help="index of the mode to color eigenvector magnitudes by",
    )
    p.add_argument(
        "-v",
        "--vmag_filepath",
        #  type=str,
        help="filepath to eigenvector magnitudes used to color mesh",
    )

    p = p.parse_args()
    path_input_mesh = p.input_filepath
    path_output = p.output_filepath
    mode_ind = int(p.mode_index)
    path_mags = p.vmag_filepath

    # delete default blender cube
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()

    # import STL
    bpy.ops.import_mesh.stl(filepath=path_input_mesh)

    # edit mode
    mode = bpy.context.active_object.mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # get mesh and setup vertex colors
    mesh = bpy.context.active_object.data
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors.active

    # load color values
    values = np.load(path_mags)[:, mode_ind]
    colors = []
    for value in values:
        colors.append([1 - value, 0., value, 1.])

    # set colors
    vertex_indices = []
    for poly in mesh.polygons:
        for v in poly.vertices:
            vertex_indices.append(v)           
    for i in range(len(color_layer.data)):
        color_layer.data[i].color = colors[vertex_indices[i]]

    # object mode
    bpy.ops.object.mode_set(mode=mode)

    # save blend file
    if os.path.exists(path_output):
        os.remove(path_output) 
    bpy.ops.wm.save_as_mainfile(filepath=path_output)


if __name__ == "__main__":
    main()
