from pytorch_lightning.loggers import TensorBoardLogger
from mmengine import Config
import datetime
from pathlib import Path


def _extract_experiment_name(cfg: Config) -> str:
    """
    Converts the config filename to a string that can be used as an experiment name.

    Because we are possibly given a path to a config file not in a path relative to
    the root of this repo, we need to deploy  some hacky logic to get the experiment name.
    """
    cfg_filename = Path(cfg.filename)

    configs_subdir = Path() / "configs"

    # If configs subdir is a parent of the config file, we can just use the path below the configs subdir
    if configs_subdir in cfg_filename.parents:
        return cfg_filename.relative_to(configs_subdir).with_suffix("")

    # If "configs" is in the config filename, we can use the path below the "configs" subdir
    if "configs" in cfg_filename.parts:
        return (
            cfg_filename.relative_to(cfg_filename.parts[cfg_filename.parts.index("configs") + 1])
            .with_suffix("")
            .as_posix()
        )

    # If "launch_files" is in the config filename, we use the name of the folders one and two levels below
    if "launch_files" in cfg_filename.parts:
        launch_files_idx = cfg_filename.parts.index("launch_files")
        return (
            cfg_filename.parts[launch_files_idx + 1]
            + "/"
            + cfg_filename.parts[launch_files_idx + 2]
        )

    return cfg_filename.absolute().with_suffix("").as_posix()


def setup_tb_logger(cfg: Config, script_name: str) -> TensorBoardLogger:

    # Save Dir
    base_logging_dir = (Path() / "tb_logs" / script_name).absolute()
    experiment_name = _extract_experiment_name(cfg)

    # Name
    version = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    tbl = TensorBoardLogger(base_logging_dir, name=experiment_name, version=version)
    print("Tensorboard logs will be saved to:", tbl.log_dir, flush=True)
    return tbl
