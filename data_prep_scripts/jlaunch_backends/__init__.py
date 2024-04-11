from .slurm import backend_slurm
from .ngc import backend_ngc
from .shared_utils import run_cmd

backends_list = [backend_slurm, backend_ngc]
