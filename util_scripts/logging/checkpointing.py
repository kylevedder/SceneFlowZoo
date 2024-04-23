from mmengine import Config
import datetime
from pathlib import Path
import os


def get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_checkpoint_path(cfg: Config) -> Path:
    checkpoint_dir_name = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    cfg_filename = Path(cfg.filename)
    config_name = cfg_filename.stem
    parent_name = cfg_filename.parent.name
    parent_path = Path(f"model_checkpoints/{parent_name}/{config_name}/")
    rank = get_rank()
    if rank == 0:
        # Since we're rank 0, we can create the directory
        return parent_path / checkpoint_dir_name
    else:
        # Since we're not rank 0, we shoulds grab the most recent directory instead of creating a new one.
        checkpoint_path = sorted(parent_path.glob("*"))[-1]
        return checkpoint_path
