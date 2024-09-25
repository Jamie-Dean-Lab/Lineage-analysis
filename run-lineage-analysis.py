#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import socket
import sys
import parsl
from parsl import bash_app

from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import GridEngineProvider
from parsl.usage_tracking.levels import LEVEL_1


# parsl.set_stream_logger(level=parsl.logging.DEBUG)


if 'myriad.ucl.ac.uk' in socket.gethostname():
    system = 'myriad.rc.ucl.ac.uk'
else:
    system = 'local'


this_dir = os.path.dirname(os.path.realpath(__file__))


def get_logdir_root(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return '/lustre/scratch/scratch/cceamgi/tmp/parsl-test'
    else:
        return os.path.join(this_dir, 'logdir')


def source_bashrc(system):
    # On some systems it may be necessary to load the shell init script to be
    # able to run commands like `module` and such.  But if it's not necessary we
    # just don't do anything.
    if system == 'myriad.rc.ucl.ac.uk':
        return f'source {str(Path.home())}/.bashrc'
    else:
        return ''


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


def module_load_julia(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return 'module load julia/1.10.1'
    else:
        return ''


def get_worker_init(system):
    return f'{source_bashrc(system)}; {module_load_python(system)}; {python_path()}; {module_load_julia(system)}'


def get_htc_executor(system):
    return HighThroughputExecutor(
        label=system,
        max_workers_per_node=1,
        worker_logdir_root=get_logdir_root(system),
        provider=GridEngineProvider(
            channel=LocalChannel(),
            nodes_per_block=1,
            init_blocks=1,
            max_blocks=1,
            walltime="00:20:00",
            scheduler_options='#$ -pe smp 8', # Input your scheduler_options if needed
            # Parsl python environment need to be loaded and activated also
            # on the compute node.
            worker_init=get_worker_init(system),
        ),
    )


# Define configuration
def get_config():
    if system == 'local':
        return None

    return Config(
        executors=[
            get_htc_executor(system),
        ],
        #  AdHoc Clusters should not be setup with scaling strategy.
        strategy='none',
        usage_tracking=LEVEL_1,
    )


@bash_app
def run_lineage_analysis(this_dir):
    return f"""julia -t auto --project={this_dir} -e 'using Pkg; Pkg.instantiate(); include("{this_dir}/controlgetlineageABCdynamics.jl"); controlgetlineageABCdynamics()'"""


with parsl.load(get_config()):
    run_lineage_analysis(this_dir).result()
