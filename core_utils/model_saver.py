from abc import ABC, abstractmethod
from pathlib import Path
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from bucketed_scene_flow_eval.datastructures import EgoLidarFlow
from core_utils import save_feather
import pandas as pd
from bucketed_scene_flow_eval.interfaces import LoaderType


class ModelOutSaver(ABC):

    def save_batch(
        self,
        input_batch: list[BucketedSceneFlowInputSequence],
        output_batch: list[BucketedSceneFlowOutputSequence],
    ):
        for input_elem, output_elem in zip(input_batch, output_batch):
            self.save(input_elem, output_elem)

    @abstractmethod
    def save(self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence):
        raise NotImplementedError()

    @abstractmethod
    def is_saved(self, input: BucketedSceneFlowInputSequence) -> bool:
        """
        Check if the output for the given input is saved already (useful for caching).
        """
        raise NotImplementedError()

    def load_saved(self, input: BucketedSceneFlowInputSequence) -> BucketedSceneFlowOutputSequence:
        """
        Load the saved output for the given input.
        """
        raise NotImplementedError()

    def load_saved_batch(
        self, input_batch: list[BucketedSceneFlowInputSequence]
    ) -> list[BucketedSceneFlowOutputSequence]:
        return [self.load_saved(input) for input in input_batch]


class FlowNoSave(ModelOutSaver):
    def save(self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence):
        pass

    def is_saved(self, input: BucketedSceneFlowInputSequence) -> bool:
        return False


class FlowSave(ModelOutSaver):

    def __init__(self, save_root: Path) -> None:
        if isinstance(save_root, str):
            save_root = Path(save_root)
        assert isinstance(save_root, Path), f"Expected Path, but got {type(save_root)}"
        self.save_root = save_root

    def _save_flow(self, save_path: Path, ego_flow: EgoLidarFlow):
        """
        Save each flow as a single result.

        This assumes that the output is length is 1.

        The content of the feather file is a dataframe with the following columns:
         - is_valid
         - flow_tx_m
         - flow_ty_m
         - flow_tz_m
        """
        output_df = pd.DataFrame(
            {
                "is_valid": ego_flow.mask,
                "flow_tx_m": ego_flow.full_flow[:, 0],
                "flow_ty_m": ego_flow.full_flow[:, 1],
                "flow_tz_m": ego_flow.full_flow[:, 2],
            }
        )
        save_feather(
            save_path,
            output_df,
            verbose=False,
        )

    def _load_flow(self, save_path: Path) -> EgoLidarFlow:
        assert save_path.exists(), f"Expected {save_path} to exist, but it does not."
        output_df = pd.read_feather(save_path)
        return EgoLidarFlow(
            full_flow=output_df[["flow_tx_m", "flow_ty_m", "flow_tz_m"]].to_numpy(),
            mask=output_df["is_valid"].to_numpy(),
        )

    def _single_save_file(self, input: BucketedSceneFlowInputSequence) -> Path:
        return (
            Path(self.save_root)
            / f"sequence_len_{len(input):03d}"
            / f"{input.sequence_log_id}"
            / f"{input.dataset_idx:010d}.feather"
        )

    def _multi_save_files(self, input: BucketedSceneFlowInputSequence) -> list[Path]:
        return [
            Path(self.save_root)
            / f"{input.loader_type}"
            / f"sequence_len_{len(input):03d}"
            / f"{input.sequence_log_id}"
            / f"{input.dataset_idx:06d}"
            / f"{idx:010d}.feather"
            for idx in range(len(input) - 1)
        ]

    def _save_single_flow(
        self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence
    ):

        ego_flows = output.to_ego_lidar_flow_list()
        assert len(ego_flows) == 1, f"Expected a single ego flow, but got {len(ego_flows)}"
        self._save_flow(self._single_save_file(input), ego_flows[0])

    def _save_multi_flow(
        self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence
    ):
        ego_flows = output.to_ego_lidar_flow_list()
        save_paths = self._multi_save_files(input)
        assert (
            len(ego_flows) == len(input) - 1
        ), f"Expected {len(input) - 1} ego flows, but got {len(ego_flows)}"
        assert len(save_paths) == len(
            ego_flows
        ), f"Expected {len(ego_flows)} save paths, but got {len(save_paths)}"
        for ego_flow, save_path in zip(ego_flows, save_paths):
            self._save_flow(save_path, ego_flow)

    def _is_saved_multi(self, input: BucketedSceneFlowInputSequence) -> bool:
        return all([save_path.exists() for save_path in self._multi_save_files(input)])

    def _is_saved_single(self, input: BucketedSceneFlowInputSequence) -> bool:
        return self._single_save_file(input).exists()

    def save(self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence):
        assert isinstance(
            input, BucketedSceneFlowInputSequence
        ), f"Expected BucketedSceneFlowInputSequence, got {type(input)}"
        assert isinstance(
            output, BucketedSceneFlowOutputSequence
        ), f"Expected BucketedSceneFlowOutputSequence, got {type(output)}"
        match input.loader_type:
            case LoaderType.CAUSAL:
                self._save_single_flow(input, output)
            case LoaderType.NON_CAUSAL:
                self._save_multi_flow(input, output)
            case _:
                raise ValueError(f"Unknown loader type: {input.loader_type}")

    def is_saved(self, input: BucketedSceneFlowInputSequence) -> bool:
        match input.loader_type:
            case LoaderType.CAUSAL:
                return self._is_saved_single(input)
            case LoaderType.NON_CAUSAL:
                return self._is_saved_multi(input)
            case _:
                raise ValueError(f"Unknown loader type: {input.loader_type}")

    def _load_single_flow(self, input: BucketedSceneFlowInputSequence) -> EgoLidarFlow:
        return self._load_flow(self._single_save_file(input))

    def _load_multi_flow(self, input: BucketedSceneFlowInputSequence) -> list[EgoLidarFlow]:
        return [self._load_flow(save_path) for save_path in self._multi_save_files(input)]

    def load_saved(self, input: BucketedSceneFlowInputSequence) -> BucketedSceneFlowOutputSequence:
        _, PadN, _ = input.full_pc.shape
        match input.loader_type:
            case LoaderType.CAUSAL:
                return BucketedSceneFlowOutputSequence.from_ego_lidar_flow_list(
                    [self._load_single_flow(input)], max_len=PadN
                )
            case LoaderType.NON_CAUSAL:
                return BucketedSceneFlowOutputSequence.from_ego_lidar_flow_list(
                    self._load_multi_flow(input), max_len=PadN
                )
            case _:
                raise ValueError(f"Unknown loader type: {input.loader_type}")
