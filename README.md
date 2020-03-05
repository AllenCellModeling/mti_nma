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
```

### Blender Visualization
Some visualizations included in this    pipeline will run Blender code, requiring
the user to have Blender downloaded on their machine. You can download Blender
for free here:

https://www.blender.org/download/

The pipeline currently has the Blender path set to the default Mac OS location.
If you are using another operating system or have downloaded Blender in another
location, you will need to pass the path to where you have downloaded Blender.
An example is provided below with the example run line in the "Running" section.

## Organization
- Global config settings are in `.config`
- Individual project steps are in `mti_mnist/steps/<step_name>`

## Running

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

### Passing parameters, including required Blender path
There are several parameters you may want to pass (number of samples),
and one which you MUST pass - the path to your blender application
(if you are not running on a Mac with Blender downloaded in the
default location).
To pass this (or other parameters) you would run a line like the following"
- Generic parameter pass
    - `mti_nma all run --<parameter_name> <parameter_value>`
- Blender path pass
    - `mti_nma all run --path_blender <your_blender_app_path>`
- Blender path and number of samples pass
    - `mti_nma all run --nsamples <int> --path_blender <your_blender_app_path>`

### Multiprocessing

#### Default
By default, all of the actual computation (per FOV, per Cell, etc) run in parallel using either Dask, or ThreadPools.
What this means is that when you run: `mti_nma all run`, it will use all of the cores on your current machine for
processing where each core will be a Dask worker.

#### Debugging and Threading
As tracing errors with Dask can be a bit tricky, if you want to turn this off, simply run the step individually, or use
the `debug` flag (`mti_nma all run --debug`) and the processing will happen on threads instead of Dask workers.

#### Cluster
If you want to utilize the SLURM cluster with Dask you can provide the `distributed` flag
(`mti_nma all run --distributed`). However, if you are doing this you will need to tell the pipeline where `blender` is
located as well. A full command for this would be:

```bash
mti_nma all run --path_blender /allen/aics/modeling/jacksonb/applications/blender-2.82-linux64/blender --distributed
```

#### Logs and Dask Monitoring
If you aren't running in debug mode, following the processing is a bit more difficult to do. When the pipeline starts
however, a message should be printed to `log.info` that says something like:

```bash
[INFO: 121 2020-03-05 11:21:49,236] Dask dashboard available at: http://127.0.0.1:8787/status
```

If you are running on your local machine, then great, that link will simply work for you and you open up that page
to monitor what Dask is working on.

If you are running on the cluster, that link will be for the machine you are connected to in the cluster and you should
run a port forwarding command to connect to view the Dask monitor on your local machine.

I.E. In a new terminal:

```bash
ssh -A -J slurm-master -L {port_provided_by_log}:{node_you_are_on}:{port_provided_by_log} {node_you_are_on}
```

Additionally, logs every worker started by the pipeline will be placed in the directory:
`~/.dask_logs/mti_nma/{datetime_of_pipeline_start}/`

#### A Near Production Setup

The following commands are the minimal set of commands to get setup for running the pipeline with a large `nsamples`:

1. Connect to slurm-master and request a CPU node to start the Dask scheduler / run the pipeline on

    ```bash
    ssh slurm-master
    srun -c 8 -p aics_cpu_general --pty bash
    ```

2. Navigate to your `mti_nma` project folder

    ```bash
    cd projects/mti_nma
    ```

3. Setup or connect to your conda environment

    1. Setup a conda environment

        ```bash
        conda create --name mti_nma python=3.7 -y
        conda activate mti_nma
        pip install -e .
        ```

    OR:

    2. Connect to your existing conda environment

        ```bash
        conda activate mti_nma
        ```

4. Run the pipeline

    ```bash
    mti_nma all run --path_blender /allen/aics/modeling/jacksonb/applications/blender-2.82-linux64/blender --nsamples {int} --distributed
    ```

5. (Optional) Connect to the Dask UI

    _In a new terminal_

    ```bash
    ssh -A -J slurm-master -L {port_provided_by_log}:{node_you_are_on}:{port_provided_by_log} {node_you_are_on}
    ```
