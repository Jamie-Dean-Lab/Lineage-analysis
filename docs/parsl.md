# Running Parsl pipeline on a cluster

[Parsl](https://parsl-project.org/) is a workflow manager, which allows running complex scientific pipelines on a wide variety of systems.
We use Parsl to run the lineage analysis pipeline on computing clusters.

When run on a supported cluster, this tool automatically submits a job to the system queue.

> [!NOTE]
> This pipeline is expected to be installed on the same system where you want to run the pipeline.
> Remote execution (launching the pipeline from one machine to be executed on a different one) is not currently supported.

## Requirements

* [git](https://git-scm.com/)
* [Python](https://www.python.org/) 3.10 or later version

This software is typically already available on most computing clusters, refer to the documentation of the system you are using.

## Installation

Clone the repository with

```
git clone https://github.com/Jamie-Dean-Lab/Lineage-analysis.git
```

Move inside the cloned repository and install the dependencies:

```
cd Lineage-analysis
pip install -e .
```

## Running a pipeline

You can use the [`run-lineage-analysis.py`](../run-lineage-analysis.py) script to run the analysis.
The script takes two mandatory arguments:

```sh
./run-lineage-analysis.py --model <MODEL_NAME> <INPUT_FILE>
```

where `<MODEL_NAME>` is the name of the model to use for the analysis (it must be one of `"RW_FW"`, `"perfect_GE"`, `"perfect_FW"`, `"clock_FW"`, `"2DRW_FW"`, `"2DRW_GE"`, `"2DRW_F"`, `"clock_GE"`, `"RW_GE"`) and `<INPUT_FILE>` is the path to the input file you want to analyse.

For more information about all the options accepted by the script, run

```sh
./run-lineage-analysis.py --help
```

## Instructions for developers

### Adding support for new clusters

If you want to add support for a new cluster, you may need to edit the [`run-lineage-analysis.py`](../run-lineage-analysis.py) script in a few places:

* the `source_bashrc` function, to specify the command to source the bashrc script, if necessary;
* the `module_load_python` function, to specify the command to load the python module on the system;
* the `module_load_julia` function, to specify the command to load the julia module on the system, if available.  If not, juliaup will be automatically installed and there's no need to do anything else;
* the `get_scheduler_options` function if it is necessary to set extra options for the scheduler of the system;
* the `get_htc_executor` function to set the type of the executor to use on the system;
* the `main` function to automatically set the function name based on the hostname.
