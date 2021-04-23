import imageio
import os

# Steps to generate GIF of first nine modes
# Edit filepaths in mti_nma/bin/view_modes.pvsm to match your own filepaths
# In Paraview: File> Load State
# Select view_modes.pvsm located in mti_nma/bin/
# At Load State Data File Options "Use file names from state"
# You should now see the 9-grid of modes
# To save pngs of frames: File> Save Animation
# Save in whatever directory you'd like, and use this as "pngdir" below
# In the base mti_nma directory for this repo, run "python make_9grid_mode_video.png"
# The gif should now be saved in your specified save_file_path below


def create_gif_from_png_dir(
        pngdir="../../local_staging/nma/nma_Nuc/nma_data/9grid_frames/",
        save_file_path="../../local_staging/nma/nma_Nuc/nma_data/9grid.gif"):

    # Create empty array 
    images = []
    for filename in os.listdir(pngdir):
        images.append(imageio.imread(pngdir + filename))
    imageio.mimsave(save_file_path, images)


create_gif_from_png_dir()
