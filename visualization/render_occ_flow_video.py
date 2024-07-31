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
from visualization.vis_lib import ColorEnum
from dataclasses import dataclass
import numpy as np
import torch
from models.components.neural_reps import (
    GigaChadOccFlowMLP,
    SimpleEncoder,
    FourierTemporalEmbedding,
    ActivationFn,
)
from models.mini_batch_optimization import GigachadOccFlowModel
from models.components.neural_reps import ModelOccFlowResult, QueryDirection
import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visualization.flow_to_rgb import flow_to_rgb
from models.whole_batch_optimization.checkpointing.model_loader import OptimCheckpointModelLoader
from models import ForwardMode


@dataclass
class GridSampler:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    x: np.ndarray
    y: np.ndarray
    xys: np.ndarray
    xy_idxes: np.ndarray

    @staticmethod
    def from_torch_grid(torch_input_sequence: TorchFullFrameInputSequence) -> "GridSampler":
        min_x = torch.inf
        max_x = -torch.inf
        min_y = torch.inf
        max_y = -torch.inf
        for idx in range(len(torch_input_sequence)):
            points = torch_input_sequence.get_global_pc(idx)
            min_x = min(min_x, points[:, 0].min().item())
            max_x = max(max_x, points[:, 0].max().item())
            min_y = min(min_y, points[:, 1].min().item())
            max_y = max(max_y, points[:, 1].max().item())

        x = np.arange(min_x, max_x, 0.2)
        y = np.arange(min_y, max_y, 0.2)
        x_idxes = np.arange(len(x))
        y_idxes = np.arange(len(y))

        xy_grid = np.array(np.meshgrid(x, y))
        xy_idx_grid = np.array(np.meshgrid(x_idxes, y_idxes))

        xys = xy_grid.T.reshape(-1, 2)
        xy_idxes = xy_idx_grid.T.reshape(-1, 2)

        return GridSampler(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            x=x,
            y=y,
            xys=xys,
            xy_idxes=xy_idxes,
        )

    def make_xyzs(self, z: float) -> np.ndarray:
        return np.concatenate([self.xys, np.full((self.xys.shape[0], 1), z)], axis=1)


class OccFlowModel:

    def __init__(
        self,
        torch_input_sequence: TorchFullFrameInputSequence,
        optimization_model: GigachadOccFlowModel,
    ):
        self.torch_input_sequence = torch_input_sequence
        self.optimization_model = optimization_model
        self.grid_sampler = GridSampler.from_torch_grid(self.torch_input_sequence)

    def inference_model(
        self,
        idx: int,
        z_value: float,
    ) -> ModelOccFlowResult:
        xyzs = self.grid_sampler.make_xyzs(z_value)
        with torch.no_grad():
            xyzs_torch = torch.tensor(xyzs, dtype=torch.float32, device="cuda")
            occ_flow_res: ModelOccFlowResult = self.optimization_model.model(
                xyzs_torch,
                idx,
                len(self.torch_input_sequence),
                QueryDirection.FORWARD,
            )
        return occ_flow_res


def load_frame_sequence(
    root_dir: Path,
    sequence_id: str,
    sequence_length: int,
) -> list[TimeSyncedSceneFlowFrame]:
    assert root_dir.is_dir(), f"Expected a directory, but got {root_dir}"
    assert isinstance(sequence_id, str), f"Expected a string, but got {sequence_id}"
    assert sequence_length > 0, f"Expected a positive integer, but got {sequence_length}"
    dataset = construct_dataset(
        name="Argoverse2NonCausalSceneFlow",
        args=dict(
            root_dir=root_dir,
            subsequence_length=sequence_length,
            with_ground=False,
            range_crop_type="ego",
            use_gt_flow=False,
            load_flow=False,
            log_subset=[sequence_id],
        ),
    )
    frame_list = dataset[0]
    return frame_list


class OccFlowVisualizer:

    def __init__(self, occ_flow_loader: OccFlowModel):
        self.model = occ_flow_loader

    def _make_flow_image(self, model_res: ModelOccFlowResult, z_value: float) -> np.ndarray:
        flow = model_res.flow.cpu().numpy()
        flow_bev_image = np.zeros(
            (len(self.model.grid_sampler.x), len(self.model.grid_sampler.y), 3), dtype=np.uint8
        )
        flow_rgbs = flow_to_rgb(flow, flow_max_radius=0.15, background="bright")
        xy_idxes = self.model.grid_sampler.xy_idxes
        flow_bev_image[xy_idxes[:, 0], xy_idxes[:, 1]] = flow_rgbs

        return flow_bev_image

    def _make_occ_image(self, model_res: ModelOccFlowResult, z_value: float) -> np.ndarray:
        occ = model_res.occ.cpu().numpy()

        # Clamp the occupancy to [0, 1]
        occ = np.clip(occ, 0, 1)
        occ_bev_image = np.zeros(
            (
                len(self.model.grid_sampler.x),
                len(self.model.grid_sampler.y),
            ),
            dtype=np.float32,
        )
        xy_idxes = self.model.grid_sampler.xy_idxes
        occ_bev_image[xy_idxes[:, 0], xy_idxes[:, 1]] = occ
        return occ_bev_image

    def _make_lidar_image(self, idx: int) -> np.ndarray:
        pc = self.model.torch_input_sequence.get_global_pc(idx).clone().cpu().numpy()
        lidar_image = np.zeros(
            (
                len(self.model.grid_sampler.x),
                len(self.model.grid_sampler.y),
            ),
            dtype=np.float32,
        )

        min_x = self.model.grid_sampler.min_x
        max_x = self.model.grid_sampler.max_x
        min_y = self.model.grid_sampler.min_y
        max_y = self.model.grid_sampler.max_y

        # Convert the lidar points to their idxes in the lidar image grid
        x_idxes = np.floor((pc[:, 0] - min_x) / (max_x - min_x) * len(self.model.grid_sampler.x))
        y_idxes = np.floor((pc[:, 1] - min_y) / (max_y - min_y) * len(self.model.grid_sampler.y))

        # Clip the idxes to the image size
        x_idxes = np.clip(x_idxes, 0, len(self.model.grid_sampler.x) - 1).astype(int)
        y_idxes = np.clip(y_idxes, 0, len(self.model.grid_sampler.y) - 1).astype(int)

        # Set the lidar points in the image
        lidar_image[x_idxes, y_idxes] = 1.0
        return lidar_image

    def _make_composite(self, occ_image: np.ndarray, flow_image: np.ndarray) -> np.ndarray:
        """
        Use the occ image as a mask to overlay the flow image
        """
        flow_image_float = flow_image.astype(np.float32) / 255
        flattened_flow_image = flow_image_float.reshape(-1, 3)
        height, width = occ_image.shape
        base_image_grayscale = occ_image.reshape(-1)
        # Turn the (N,) array into an (N, 3) array
        base_image_color = np.stack([base_image_grayscale] * 3, axis=1)
        is_occ = (occ_image > 0.5).reshape(-1)
        base_image_color[is_occ] = flattened_flow_image[is_occ]
        composite_image = base_image_color.reshape(height, width, 3)
        return composite_image

    # def visualize_occ_flow(self, z_value: float):
    #     for idx in tqdm.tqdm(range(len(self.frame_list))):
    #         occ_flow_res = self.model.inference_model(idx, z_value)
    #         lidar_image = self._make_lidar_image(idx)
    #         flow_image = self._make_flow_image(occ_flow_res, z_value)
    #         occ_image = self._make_occ_image(occ_flow_res, z_value)
    #         # TODO: Actually visualize the frames to make a movie

    def visualize_occ_flow(self, z_value: float, output_path: Path):
        # Create a figure and axes for plotting
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.reshape(-1)

        def make_images(idx: int):
            occ_flow_res = self.model.inference_model(idx, z_value)
            lidar_image = self._make_lidar_image(idx)
            flow_image = self._make_flow_image(occ_flow_res, z_value)
            occ_image = self._make_occ_image(occ_flow_res, z_value)
            composite_image = self._make_composite(occ_image.copy(), flow_image.copy())
            return lidar_image, flow_image, occ_image, composite_image

        lidar_image_zero, flow_image_zero, occ_image_zero, composite_image_zero = make_images(0)
        # Initialize image artists (placeholders for our images)
        lidar_im = axes[0].imshow(lidar_image_zero, cmap="gray")
        composite_im = axes[1].imshow(composite_image_zero)
        occ_im = axes[2].imshow(occ_image_zero, cmap="gray")
        flow_im = axes[3].imshow(flow_image_zero)

        # Set titles for each subplot
        axes[0].set_title("Lidar")
        axes[1].set_title("Composite")
        axes[2].set_title("Occupancy")
        axes[3].set_title("Flow")

        bar = tqdm.tqdm(total=len(self.model.torch_input_sequence))

        # Function to update the images for each frame
        def update_images(idx):
            lidar_image, flow_image, occ_image, composite_image = make_images(idx)

            lidar_im.set_data(lidar_image)
            composite_im.set_data(composite_image)
            occ_im.set_data(occ_image)
            flow_im.set_data(flow_image)

            bar.update(1)

            return lidar_im, composite_im  # , flow_im, occ_im

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update_images, frames=len(self.model.torch_input_sequence), interval=100, blit=True
        )  # 100 ms interval for 10 Hz

        # Save the animation
        output_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_path / "occ_flow_video.mp4"
        ani.save(output_file_path, writer="ffmpeg", fps=10)
        bar.close()
        print(f"Saved video to {output_file_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("sequence_id", type=str)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--sequence_id_to_length",
        type=Path,
        default=Path("data_prep_scripts/argo/av2_test_sizes.json"),
    )
    args = parser.parse_args()

    model_loader = OptimCheckpointModelLoader.from_checkpoint_dirs(
        args.config, args.checkpoint_root, args.sequence_id, args.sequence_id_to_length
    )
    model, full_torch_sequence = model_loader.load_model()

    visualizer = OccFlowVisualizer(OccFlowModel(full_torch_sequence, model))

    visualizer.visualize_occ_flow(z_value=0.5, output_path=args.output_path / args.sequence_id)


if __name__ == "__main__":
    main()
