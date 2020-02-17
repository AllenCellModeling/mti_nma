# mti_nma

MTI Workflow to run normal mode analysis on 3D nuclear segmentations.

This workflow currently relies on segmentations downloaded from Labkey, 
so running the pipeline as-is on Labkey data can only be done from within AICS.

---

## Installation

### Data access
If installing *somewhere other than AICS compute-cluster infrastructure* (e.g. your local machine)
... you will need:

**AICS certificates** to be able to install the required package `lkaccess`. Instructions to setup certs on an macOS machine are as follows:

- Visit http://confluence.corp.alleninstitute.org/display/SF/Certs+on+OS+X
- Download the three .crt files, open each and keychain to System and hit 'Add' to trust
- Download `pip_conf_setup.sh` to project directory
- Install wget: `brew install wget`
- Run the downloaded setup file: `sudo bash pip_conf_setup.sh`

### Normal users
Clone this repository, then
```
cd mti_nma
conda create --name mti_nma python=3.7
conda activate mti_nma
pip install -e .
```

### Developers
After following the "Normal users" installation instructions,
```
pip install -e .[all]
pre-commit install
```

### Blender Visualization
Some visualizations included in this    pipeline will run Blender code, requiring
the user to have Blender downloaded on their machine. You can download Blender
for free here:
https://www.blender.org/download/
In the file `mti_nma/mti_nma/steps/nma/nma_viz.py` you will need to update the
Blender path in `color_vertices_by_magnitude` to your own local Blender path.

## Organization
- Global config settings are in `.config`
- Individual project steps are in `mti_mnist/steps/<step_name>`

## Running

### Mount data if running locally on MacOS
Create a data directory in your local repo to mount the allen data 
```
cd mti_nma
mkdir data
```

**mount the remote data repository**, which can be done on macOS with 

```
mount_smbfs //<YOUR_USERNAME>@allen/programs/allencell/data ./data/
```

To unmount when you're all done:

```
umount ./data/
```

### Individual steps
- to run an indiviual step such as `norm`, the cmd line workflow would be
    - `mti_nma norm pull`
    - `mti_nma norm run`
    - `mti_nma norm push`

### Everything all at once
- to run the entire workflow, from the cmd line use
    - `mti_nma all pull`
    - `mti_nma all run`
    - `mti_nma all push`
