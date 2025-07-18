#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import socket
import sys

import click
import parsl
from parsl import bash_app
#from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import GridEngineProvider
from parsl.usage_tracking.levels import LEVEL_1


this_dir = os.path.dirname(os.path.realpath(__file__))


# Command to source the bashrc script on the compute nodes of the system, if
# necessary.
def source_bashrc(system):
    # On some systems it may be necessary to load the shell init script to be
    # able to run commands like `module` and such.  But if it's not necessary we
    # just don't do anything.
    if system == 'myriad.rc.ucl.ac.uk':
        return f'source {str(Path.home())}/.bashrc'
    else:
        return ''

# Command to unload any existing python/julia module 
def unload_package(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return f'module unload python\nmodule unload julia'
    else:
        return ''



# Command to run to load the Python module on the system, if necessary.
def module_load_python(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return 'module load python/3.11.3'
    else:
        return ''


# This is a hack to work around <https://github.com/Parsl/parsl/issues/3541>: we
# need to make `process_worker_pool.py` available in PATH on the compute node.
# I'd much prefer this was properly fixed upstream, but we have to do something
# ourselves here.
def python_path():

    # Parsl needs to run `process_worker_pool.py` on the compute node, how do we
    # find it?  First attempt: we try to find it in this process and we add its
    # directory to PATH.
    process_worker_pool = shutil.which('process_worker_pool.py')
    if process_worker_pool:
        return f'PATH="{os.path.dirname(process_worker_pool)}:${{PATH}}"'

    # We didn't find `process_worker_pool.py`, too bad, then let's try to add
    # `python`'s directory to PATH as a last resort, although if first attempt
    # didn't work it's unlikely this is going to help much, but at least we
    # tried something.  Note: do NOT use `os.path.realpath(sys.executable)`
    # because the `python` in the virtual environment may be a symlink to the
    # another python.  Python virtual environments are fun like that.
    return f'PATH="{os.path.dirname(sys.executable)}:${{PATH}}"'


# In case Julia isn't available in the system, install it using juliaup:
# <https://github.com/JuliaLang/juliaup>.
def install_juliaup():
    return '''
    if which julia &> /dev/null; then
        export JL_BINDIR=""
    else
        if which wget &> /dev/null; then
            DOWNLOADER="wget -O -"
        elif which curl &> /dev/null; then
            DOWNLOADER="curl -fsSL"
        else
            echo "No known downloader (wget or curl) available to install Julia"
        fi # if which wget

        ${DOWNLOADER} https://install.julialang.org | sh -s -- --yes

        export JL_BINDIR="${HOME}/.juliaup/bin/"
    fi # if which julia
    '''


# Command to load the Julia module, if available.  If not available, simply use
# juliaup to install Julia.
def module_load_julia(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return 'module load julia/1.10.1'
    else:
        # We don't know how to use julia here, let's try to install it with
        # Juliaup if not available.
        return install_juliaup()


def get_worker_init(system):
    return f'''
    {unload_package(system)}
    {module_load_python(system)}
    {source_bashrc(system)}
    {python_path()}
    {module_load_julia(system)}
    '''


def get_scheduler_options(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return f'#$ -pe smp 8\n#$ -M {os.getlogin()}@ucl.ac.uk\n#$ -m bea'
    else:
        return ''


def get_htc_executor(system, walltime, logdir):

    if system == 'myriad.rc.ucl.ac.uk':
        provider_type = GridEngineProvider

    return HighThroughputExecutor(
        label=system,
        max_workers_per_node=1,
        worker_logdir_root=logdir if logdir else os.path.join(this_dir, 'logdir'),
        provider=provider_type(
#            channel=LocalChannel(),
            nodes_per_block=1,
            init_blocks=1,
            max_blocks=1,
            walltime=walltime,
            scheduler_options=get_scheduler_options(system), # Input your scheduler_options if needed
            # Parsl python environment need to be loaded and activated also
            # on the compute node.
            worker_init=get_worker_init(system),
        ),
    )


# Define configuration
def get_config(system, walltime, logdir):
    if system == 'local':
        return None

    return Config(
        executors=[
            get_htc_executor(system, walltime, logdir),
        ],
        #  AdHoc Clusters should not be setup with scaling strategy.
        strategy='none',
        usage_tracking=LEVEL_1,
    )


@bash_app
def run_lineage_analysis(script):
    return script


# Helper function for passing the arguments to the main script: some arguments
# are of type `String`, but default to `nothing`, we want to pass `nothing`
# as-is, but otherwise pass the string with its quotes.
def nothing_or_string(str):
    if str == "nothing":
        return str
    return f"\"{str}\""


@click.command()
@click.option("--system", default="auto", help="Name of the system where to run the pipeline")
@click.option("--walltime", default="12:00:00", help="Walltime for jobs sent to the scheduler")
@click.option("--logdir", default="", help="Directory where to store the Parsl log files")
@click.option("--trunkfilename", default="nothing")
@click.option("--comment", default="nothing")
@click.option("--nochains", default="nothing", help="Number of independent chains for convergence statistic")
@click.option("--model", default="nothing")
@click.option("--timeresolution", default="nothing")
@click.option("--mcmax", default="nothing", help="Last iteration")
@click.option("--subsample", default="nothing", help="Subsampling frequency")
@click.option("--nomothersamples", default="nothing", help="Number of samples for sampling empirically from unknownmotherdistribution")
@click.option("--nomotherburnin", default="nothing", help="Burnin for sampling empirically from unknownmotherdistribution")
@click.option("--nolevels", default="nothing", help="Number of levels before posterior, first one is prior")
@click.option("--notreeparticles", default="nothing", help="Number of particles to estimate effect of nuisance parameters")
@click.option("--auxiliaryfoldertrunkname", default="nothing", help="Trunkname of folder, where auxiliary files are saved, if useRAM is 'false'")
@click.option("--useram", default="nothing", help="'true' for saving variables into workspace, 'false' for saving in external textfiles")
@click.option("--withcuda", default="nothing", help="'true' for using CUDA GPU, 'false' for without using GPU")
@click.option("--trickycells", default="nothing", help="Cells that need many particles to not lose them; in order of appearance in lineagetree")
@click.option("--without", default="nothing", help="'0' only warnings, '1' basic output, '2' detailied output, '3' debugging")
@click.option("--withwriteoutputtext", default="nothing", help="'true' if output of textfile, 'false' otherwise")
@click.option("--plotgraph", default="", help="'Read and analysis the result of the simulation'")
@click.argument("input_file")
def main(system, walltime, logdir, trunkfilename, comment, nochains, model,
         timeresolution, mcmax, subsample, nomothersamples, nomotherburnin,
         nolevels, notreeparticles, auxiliaryfoldertrunkname, useram, withcuda,
         trickycells, without, withwriteoutputtext, plotgraph, input_file):

    # If you want to support a new system, set the `system` name based on the
    # hostname.
    if system == 'auto':
        if 'myriad.ucl.ac.uk' in socket.gethostname():
            system = 'myriad.rc.ucl.ac.uk'
        else:
            system = 'local'

    # The bash script we'll eventually run
    script = ''

    if system == 'local':
        # We may need to install juliaup if we're running "locally".
        script = script + install_juliaup()

    # The actual pipeline
    script = script + f'''
    ${{JL_BINDIR}}julia -t auto --project={this_dir} -e '
    using Pkg

    Pkg.Registry.add("General") # Add registry in case we are in a fresh depot
    Pkg.Registry.update()
    Pkg.resolve()
    Pkg.instantiate()

    include("{this_dir}/controlgetlineageABCdynamics.jl"); controlgetlineageABCdynamics(;
        trunkfilename = {nothing_or_string(trunkfilename)},
        filename = {nothing_or_string(input_file)},
        comment = {nothing_or_string(comment)},
        nochains = {nochains},
        model = {nothing_or_string(model)},
        timeresolution = {timeresolution},
        MCmax = {mcmax},
        subsample = {subsample},
        nomothersamples = {nomothersamples},
        nomotherburnin = {nomotherburnin},
        nolevels = {nolevels},
        notreeparticles = {notreeparticles},
        auxiliaryfoldertrunkname = {auxiliaryfoldertrunkname},
        useRAM = {useram},
        withCUDA = {withcuda},
        trickycells = {trickycells},
        without = {without},
        withwriteoutputtext = {withwriteoutputtext},
    )'
    '''

    with parsl.load(get_config(system, walltime, logdir)):
        run_lineage_analysis(script).result()


if __name__ == '__main__':
    main()
