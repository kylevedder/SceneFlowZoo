import argparse
from pathlib import Path
from bucketed_scene_flow_eval.datastructures import (
    EgoLidarFlow,
    TimeSyncedSceneFlowFrame,
)
from bucketed_scene_flow_eval.interfaces import LoaderType
from bucketed_scene_flow_eval.datasets import (
    construct_dataset,
    AbstractDataset,
)
from dataloaders import TorchFullFrameInputSequence
from visualization.vis_lib import ColorEnum, BaseCallbackVisualizer
from dataclasses import dataclass
import numpy as np
import open3d as o3d
import torch
from models import BaseOptimizationModel
from models.mini_batch_optimization import GigachadNSFModel, GigachadOccFlowModel
from models.components.neural_reps import ModelFlowResult, ModelOccFlowResult, QueryDirection
import tqdm


@dataclass
class FlowLoader:
    flow_folder: Path | None = None

    def __len__(self):
        return 1

    def load_ego_flow(
        self, dataset_frame_list: list[TimeSyncedSceneFlowFrame], flow_idx: int
    ) -> list[EgoLidarFlow]:
        raise NotImplementedError("Must implement load_ego_flow method")

    def load_occupancy_grid(
        self, dataset_frame_list: list[TimeSyncedSceneFlowFrame]
    ) -> np.ndarray | None:
        return None


@dataclass(kw_only=True)
class ModelWeightsFlowLoader(FlowLoader):
    model_checkpoints: list[Path]
    model_type: type[GigachadNSFModel] = GigachadNSFModel

    def __post_init__(self):
        assert len(self.model_checkpoints) > 0, "No model checkpoints found"

    def __len__(self):
        return len(self.model_checkpoints)

    def _setup_model(
        self, dataset_frame_list: list[TimeSyncedSceneFlowFrame], flow_idx: int
    ) -> tuple[GigachadNSFModel, TorchFullFrameInputSequence]:
        torch_input_sequence = TorchFullFrameInputSequence.from_frame_list(
            0, dataset_frame_list, 120000, LoaderType.NON_CAUSAL
        )

        assert len(self.model_checkpoints) > 0, "No model checkpoints found"
        assert flow_idx < len(self), f"Flow index {flow_idx} out of bounds"

        checkpoint_path = self.model_checkpoints[flow_idx]
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_dict = torch.load(checkpoint_path)
        print(f"Loaded checkpoint")
        model_weights = checkpoint_dict["model"]
        model = self.model_type(
            full_input_sequence=torch_input_sequence,
            speed_threshold=60.0 / 10.0,
            pc_target_type="lidar_camera",
            pc_loss_type="forward_only",
        )
        model.load_state_dict(model_weights)
        model.eval()
        model = model.to("cuda")
        torch_input_sequence = torch_input_sequence.to("cuda")

        return model, torch_input_sequence

    def load_ego_flow(
        self, dataset_frame_list: list[TimeSyncedSceneFlowFrame], flow_idx: int
    ) -> list[EgoLidarFlow]:
        model, torch_input_sequence = self._setup_model(dataset_frame_list, flow_idx)

        with torch.no_grad():
            model_output = model.inference_forward_single(torch_input_sequence, None)

        return model_output.to_ego_lidar_flow_list()


@dataclass
class ModelWeightsOccFlowLoader(ModelWeightsFlowLoader):
    model_type: type[GigachadOccFlowModel] = GigachadOccFlowModel

    def load_ego_flow(
        self, dataset_frame_list: list[TimeSyncedSceneFlowFrame], flow_idx: int
    ) -> list[EgoLidarFlow]:
        model, torch_input_sequence = self._setup_model(dataset_frame_list, flow_idx)

        with torch.no_grad():
            model_output = model.inference_forward_single(torch_input_sequence, None)

        self._load_occupancy_grid(model, dataset_frame_list)

        return model_output.to_ego_lidar_flow_list()

    def _load_occupancy_grid(
        self,
        model: GigachadOccFlowModel,
        dataset_frame_list: list[TimeSyncedSceneFlowFrame],
        z: float = 0.5,  # meters
    ):

        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        for item in dataset_frame_list:
            points = item.pc.global_pc.points
            min_x = min(min_x, points[:, 0].min())
            max_x = max(max_x, points[:, 0].max())
            min_y = min(min_y, points[:, 1].min())
            max_y = max(max_y, points[:, 1].max())

        # List of points at 0.1m resolution
        x = np.arange(min_x, max_x, 0.1)
        y = np.arange(min_y, max_y, 0.1)

        x_idxes = np.arange(len(x))
        y_idxes = np.arange(len(y))

        xy_grid = np.array(np.meshgrid(x, y))
        xy_idx_grid = np.array(np.meshgrid(x_idxes, y_idxes))

        xys = xy_grid.T.reshape(-1, 2)
        xy_idxes = xy_idx_grid.T.reshape(-1, 2)

        def make_img(z: float, idx: int):
            xyzs = np.concatenate([xys, np.full((xys.shape[0], 1), z)], axis=1)
            with torch.no_grad():
                xyzs_torch = torch.tensor(xyzs, dtype=torch.float32, device="cuda")
                occ_flow_res: ModelOccFlowResult = model.model(
                    xyzs_torch,
                    idx,
                    len(dataset_frame_list),
                    QueryDirection.FORWARD,
                )
            occupancy_bev_image = np.zeros((len(x), len(y)))
            occupancy_bev_image[xy_idxes[:, 0], xy_idxes[:, 1]] = occ_flow_res.occ.cpu().numpy()
            return occupancy_bev_image

        idxes = list(range(len(dataset_frame_list) - 1))[::10]
        zs = [0, 0.5, 1.0, 1.5, 2.0]

        import matplotlib.pyplot as plt

        # Make zs x idxes subplots
        fig, axes = plt.subplots(len(idxes), len(zs))

        bar = tqdm.tqdm(total=len(idxes) * len(zs))
        for idxidx, idx in enumerate(idxes):
            for zidx, z in enumerate(zs):
                axes[idxidx, zidx].imshow(make_img(z, idx))
                axes[idxidx, zidx].set_title(f"Z: {z}m, idx: {idx}")
                bar.update(1)
        bar.close()
        # make tight
        plt.tight_layout()
        plt.show()


@dataclass
class SavedFlowLoader(FlowLoader):

    def load_ego_flow(
        self, dataset_frame_list: list[TimeSyncedSceneFlowFrame], flow_idx: int
    ) -> list[EgoLidarFlow]:
        return [frame.flow for frame in dataset_frame_list[:-1]]


@dataclass(kw_only=True)
class SubsequenceInfo:
    start_idx: int
    size: int

    def __repr__(self) -> str:
        return f"Start: {self.start_idx} Size: {self.size}"


@dataclass(kw_only=True)
class VisState:
    flow_color: ColorEnum = ColorEnum.RED
    intermedary_result_idx: int = 0
    subsequence: SubsequenceInfo | None = None
    with_auxillary_pc: bool = True

    def __repr__(self) -> str:
        ret_str = f"Vis State: Opt Step {self.intermedary_result_idx}"
        if self.subsequence is not None:
            ret_str += f" SubSubsequence: ({self.subsequence})"
        return ret_str


class ResultsVisualizer(BaseCallbackVisualizer):

    def __init__(
        self,
        dataset: AbstractDataset,
        flow_loader: FlowLoader,
        sequence_length: int,
        subsequece_length: int | None,
        with_auxillary_pc: bool,
        checkpoint_idx: int | None = None,
    ):

        self.dataset = dataset
        self.dataset_idx = 0
        self.subsequence_length = sequence_length
        self.flow_loader = flow_loader

        intermedary_result_idx = (
            checkpoint_idx if checkpoint_idx is not None else len(flow_loader) - 1
        )

        self.vis_state = VisState(
            intermedary_result_idx=intermedary_result_idx, with_auxillary_pc=with_auxillary_pc
        )
        if subsequece_length is not None:
            self.vis_state.subsequence = SubsequenceInfo(
                start_idx=0,
                size=subsequece_length,
            )

        super().__init__(
            screenshot_path=Path()
            / "screenshots_intermediary"
            / f"dataset_idx_{self.dataset_idx:010d}"
        )

    def _load_raw_frames_with_caching(self) -> list[TimeSyncedSceneFlowFrame]:
        # Check to see if self._cached_dataset_idx is the same as self.dataset_idx.
        # The field will not exist if it has not been set yet.
        if hasattr(self, "_cached_dataset_idx") and self._cached_dataset_idx == self.dataset_idx:
            assert hasattr(
                self, "_cached_dataset_frame_list"
            ), "Cached dataset frame list not found"
            return self._cached_dataset_frame_list

        assert len(self.dataset) > 0, "No sequences found in dataset"
        print(f"Datset has {len(self.dataset)} sequences")
        dataset_frame_list = self.dataset[0]

        self._cached_dataset_idx: int = self.dataset_idx
        self._cached_dataset_frame_list: list[TimeSyncedSceneFlowFrame] = dataset_frame_list

        return dataset_frame_list

    def _load_raw_flows_with_caching(self) -> list[EgoLidarFlow]:
        if (
            hasattr(self, "_cached_flows_idx")
            and self._cached_flows_idx == self.vis_state.intermedary_result_idx
        ):
            assert hasattr(self, "_cached_flows"), "Cached flows not found"
            return self._cached_flows

        # Flows need to be loaded, they are not cached.
        dataset_frame_list = self._load_raw_frames_with_caching()
        intermediary_ego_flows = self.flow_loader.load_ego_flow(
            dataset_frame_list, self.vis_state.intermedary_result_idx
        )

        self._cached_flows_idx: int = self.vis_state.intermedary_result_idx
        self._cached_flows: list[EgoLidarFlow] = intermediary_ego_flows

        return intermediary_ego_flows

    def _load_frame_list(self) -> list[TimeSyncedSceneFlowFrame]:
        dataset_frame_list = self._load_raw_frames_with_caching()
        intermediary_ego_flows = self._load_raw_flows_with_caching()

        assert (
            len(dataset_frame_list) == len(intermediary_ego_flows) + 1
        ), f"Expected one more frame in dataset than intermediary flows; instead found {len(dataset_frame_list)} and {len(intermediary_ego_flows)}"

        for frame_info, ego_flow in zip(dataset_frame_list, intermediary_ego_flows):
            # Add the intermediary ego flow to the frame info
            frame_info.flow = ego_flow

        return dataset_frame_list

    def _get_screenshot_path(self) -> Path:
        file_stem = f"{self.vis_state.intermedary_result_idx:08d}"
        if self.vis_state.subsequence is not None:
            file_stem += f"_subsubsequence_{self.vis_state.subsequence.start_idx:08d}"
        return self.screenshot_path / f"{file_stem}.png"

    def _print_instructions(self):
        print("#############################################################")
        print("Flow moves from the gray point cloud to the white point cloud\n")
        print("Press up or down arrow to change intermediary result")
        print(f"Press S to save screenshot (saved to {self.screenshot_path.absolute()})")
        print("Press E to jump to end of sequence")
        print("Press A to toggle auxillary point cloud")
        print("#############################################################")

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        super()._register_callbacks(vis)
        # up arrow increase result_idx
        vis.register_key_callback(265, self.increase_result_idx)
        # down arrow decrease result_idx
        vis.register_key_callback(264, self.decrease_result_idx)
        # right arrow increase subsubsequence_idx
        vis.register_key_callback(262, self.increase_subsubsequence_idx)
        # left arrow decrease subsubsequence_idx
        vis.register_key_callback(263, self.decrease_subsubsequence_idx)
        # E to jump to end of sequence
        vis.register_key_callback(ord("E"), self.jump_to_end_of_sequence)
        # A to toggle auxillary point cloud
        vis.register_key_callback(ord("A"), self.toggle_auxillary_pc)

    def toggle_auxillary_pc(self, vis):
        self.vis_state.with_auxillary_pc = not self.vis_state.with_auxillary_pc
        self.draw_everything(vis, reset_view=False)

    def increase_result_idx(self, vis):
        self.vis_state.intermedary_result_idx += 1
        if self.vis_state.intermedary_result_idx >= len(self.flow_loader):
            self.vis_state.intermedary_result_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_result_idx(self, vis):
        self.vis_state.intermedary_result_idx -= 1
        if self.vis_state.intermedary_result_idx < 0:
            self.vis_state.intermedary_result_idx = len(self.flow_loader) - 1
        self.draw_everything(vis, reset_view=False)

    def increase_subsubsequence_idx(self, vis):
        if self.vis_state.subsequence is None:
            return
        self.vis_state.subsequence.start_idx += 1
        end_idx = self.vis_state.subsequence.start_idx + self.vis_state.subsequence.size
        if end_idx >= self.subsequence_length:
            self.vis_state.subsequence.start_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_subsubsequence_idx(self, vis):
        if self.vis_state.subsequence is None:
            return
        self.vis_state.subsequence.start_idx -= 1
        if self.vis_state.subsequence.start_idx < 0:
            self.vis_state.subsequence.start_idx = (
                self.subsequence_length - self.vis_state.subsequence.size
            )
        self.draw_everything(vis, reset_view=False)

    def jump_to_end_of_sequence(self, vis):
        self.vis_state.intermedary_result_idx = len(self.flow_loader) - 1
        self.draw_everything(vis, reset_view=False)

    def draw_everything(self, vis, reset_view=False):
        self.geometry_list.clear()
        print(f"Vis State: {self.vis_state}")
        frame_list = self._load_frame_list()
        color_list = self._frame_list_to_color_list(len(frame_list))

        if self.vis_state.subsequence is not None:
            start_idx = self.vis_state.subsequence.start_idx
            size = self.vis_state.subsequence.size
            frame_list = frame_list[start_idx : start_idx + size]
            color_list = color_list[start_idx : start_idx + size]

        elements = list(enumerate(zip(frame_list, color_list)))
        for idx, (flow_frame, color) in elements:
            pc = flow_frame.pc.global_pc
            self.add_pointcloud(pc, color=(0, 1, 0))
            if self.vis_state.with_auxillary_pc:
                aux_pc = flow_frame.auxillary_pc.global_pc
                self.add_pointcloud(aux_pc, color=(0, 0, 1))
            flowed_pc = flow_frame.pc.flow(flow_frame.flow).global_pc

            draw_color = self.vis_state.flow_color.rgb

            # Add flowed point cloud
            if (draw_color is not None) and (flowed_pc is not None) and (idx < len(frame_list) - 1):
                self.add_lineset(pc, flowed_pc, color=draw_color)
        vis.clear_geometries()
        self.render(vis, reset_view=reset_view)


def get_flow_loader(
    checkpoint_folder: Path | None, checkpoint_type: str, flow_folder: Path | None
) -> FlowLoader:

    # If both are none or both are not none, raise an error
    if checkpoint_folder is not None and flow_folder is not None:
        raise ValueError("Cannot provide both checkpoint folder and flow folder")

    if checkpoint_folder is None and flow_folder is None:
        raise ValueError("Must provide either checkpoint folder or flow folder")

    if checkpoint_folder is not None:
        # If the results folder is a particular file, then grab the parent directory
        if checkpoint_folder.is_file():
            checkpoint_folder = checkpoint_folder.parent

        if checkpoint_type == "occ_flow":
            return ModelWeightsOccFlowLoader(
                model_checkpoints=sorted(checkpoint_folder.glob("*.pth"))
            )
        elif checkpoint_type == "flow":
            return ModelWeightsFlowLoader(model_checkpoints=sorted(checkpoint_folder.glob("*.pth")))
        else:
            raise ValueError(f"Invalid checkpoint type {checkpoint_type}")
    if flow_folder is not None:
        return SavedFlowLoader(flow_folder)

    raise ValueError("Invalid flow type")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Argoverse2NonCausalSceneFlow")
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--subsequence_length", type=int, default=-1)
    parser.add_argument("--sequence_id", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=Path, default=None)
    parser.add_argument("--checkpoint_type", type=str, choices=["occ_flow", "flow"], default="flow")
    parser.add_argument("--checkpoint_idx", type=int, default=None)
    parser.add_argument("--flow_folder", type=Path, default=None)
    parser.add_argument("--with_auxillary_pc", action="store_true")
    args = parser.parse_args()

    flow_loader = get_flow_loader(
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_type=args.checkpoint_type,
        flow_folder=args.flow_folder,
    )

    sequence_length = args.sequence_length
    log_subset = [args.sequence_id] if args.sequence_id is not None else None

    dataset = construct_dataset(
        name=args.dataset_name,
        args=dict(
            root_dir=args.root_dir,
            subsequence_length=sequence_length,
            with_ground=False,
            range_crop_type="ego",
            use_gt_flow=False,
            log_subset=log_subset,
            flow_data_path=flow_loader.flow_folder,
        ),
    )
    assert len(dataset) == 1, f"Expected only one sequence, but found {len(dataset)} sequences"
    subsequence_length = args.subsequence_length if args.subsequence_length > 0 else None

    visualizer = ResultsVisualizer(
        dataset,
        flow_loader,
        sequence_length,
        subsequence_length,
        args.with_auxillary_pc,
        args.checkpoint_idx,
    )

    visualizer.run()


if __name__ == "__main__":
    main()
