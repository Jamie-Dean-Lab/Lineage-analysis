#!/usr/bin/env python3

import os
from pathlib import Path
import parsl
from parsl import bash_app

from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import GridEngineProvider
from parsl.usage_tracking.levels import LEVEL_1


# parsl.set_stream_logger(level=parsl.logging.DEBUG)

logdir_root = '/lustre/scratch/scratch/cceamgi/tmp/parsl-test'

def source_bashrc():
    return f'source {str(Path.home())}/.bashrc'


def module_load_python(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return 'module load python/3.11.3'
    else:
        return ''


def activate_venv():
    return 'source /lustre/scratch/scratch/cceamgi/repo/Lineage-analysis/env/bin/activate'


def module_load_julia(system):
    if system == 'myriad.rc.ucl.ac.uk':
        return 'module load julia/1.10.1'
    else:
        return ''

def get_worker_init(system):
    return f'{source_bashrc()}; {module_load_python(system)}; {activate_venv()}; {module_load_julia(system)}'


system = 'myriad.rc.ucl.ac.uk'

# Define configuration for Myriad
config = Config(
    executors=[
        HighThroughputExecutor(
            label=system,
            max_workers_per_node=1,
            worker_logdir_root=logdir_root,
            provider=GridEngineProvider(
                channel=LocalChannel(),
                nodes_per_block=1,
                init_blocks=1,
                max_blocks=1,
                walltime="00:30:00",
                scheduler_options='#$ -pe mpi 18', # Input your scheduler_options if needed
                # Parsl python environment need to be loaded and activated also
                # on the compute node.
                worker_init=get_worker_init(system),
            ),
        )
    ],
    #  AdHoc Clusters should not be setup with scaling strategy.
    strategy='none',
    usage_tracking=LEVEL_1,
)


this_dir = os.path.dirname(os.path.realpath(__file__))


@bash_app
def run_lineage_analysis(this_dir):
    return f"""julia -t 18 --project={this_dir} -e 'using Pkg; Pkg.instantiate(); include("{this_dir}/controlgetlineageABCdynamics.jl"); controlgetlineageABCdynamics()'"""


parsl.load(config)
run_lineage_analysis(this_dir).result()
