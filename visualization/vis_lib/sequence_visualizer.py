from bucketed_scene_flow_eval.datastructures import O3DVisualizer, TimeSyncedSceneFlowFrame
import open3d as o3d
from bucketed_scene_flow_eval.interfaces import (
    AbstractSequenceLoader,
    AbstractAVLidarSequence,
)
from .lazy_frame_matrix import EagerFrameMatrix

from dataclasses import dataclass
from .lazy_frame_matrix import AbstractFrameMatrix
from pathlib import Path
import enum
from typing import Optional
from visualization.vis_lib import BaseCallbackVisualizer


class ColorEnum(enum.Enum):
    RED = (1, 0, 0)
    GREEN = (0, 1, 0)
    BLUE = (0, 0, 1)
    DISABLED = None

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.name.lower()

    @staticmethod
    def from_str(color_name: str):
        return getattr(ColorEnum, color_name.upper())

    @property
    def rgb(self) -> Optional[tuple[float, float, float]]:
        return self.value

    def cycle(self):
        match self:
            case ColorEnum.RED:
                return ColorEnum.GREEN
            case ColorEnum.GREEN:
                return ColorEnum.BLUE
            case ColorEnum.BLUE:
                return ColorEnum.DISABLED
            case ColorEnum.DISABLED:
                return ColorEnum.RED


@dataclass(kw_only=True)
class VisState:
    full_frame_matrix: AbstractFrameMatrix
    sequence_idx: int
    frame_idx: int
    flow_color: ColorEnum = ColorEnum.RED
    frame_step_size: int = 1

    def __str__(self):
        return f"sequence_idx: {self.sequence_idx}, frame__idx: {self.frame_idx}, "


class SequenceVisualizer(BaseCallbackVisualizer):

    def __init__(
        self,
        name_sequence_list: list[tuple[str, list[TimeSyncedSceneFlowFrame]]],
        sequence_id: str,
        frame_idx: int = 0,
        subsequence_length: int = 2,
        point_size: float = 0.1,
        step_size: int = 1,
    ):
        super().__init__(point_size=point_size)
        self.sequence_id = sequence_id
        self.name_lst = [name for name, _ in name_sequence_list]
        sequence_lst = [sequence for _, sequence in name_sequence_list]
        self.vis_state = VisState(
            full_frame_matrix=EagerFrameMatrix(
                sequences=sequence_lst, subsequence_length=subsequence_length
            ),
            frame_idx=frame_idx,
            frame_step_size=step_size,
            sequence_idx=0,
        )

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        vis.register_key_callback(ord("S"), self.save_screenshot)
        vis.register_key_callback(ord("F"), self.toggle_flow_lines)
        vis.register_key_callback(ord("C"), self.save_camera_pose)
        vis.register_key_callback(ord("L"), self.load_camera_pose)
        # left arrow decrease starter_idx
        vis.register_key_callback(263, self.decrease_frame_idx)
        # right arrow increase starter_idx
        vis.register_key_callback(262, self.increase_frame_idx)
        # up arrow increase sequence_idx
        vis.register_key_callback(265, self.increase_sequence_idx)
        # down arrow decrease sequence_idx
        vis.register_key_callback(264, self.decrease_sequence_idx)

    def get_current_method_name(self) -> str:
        return self.name_lst[self.vis_state.sequence_idx]

    def _get_screenshot_path(self) -> Path:
        return (
            self.screenshot_path
            / self.sequence_id
            / f"{self.vis_state.frame_idx:06d}_{self.get_current_method_name()}.png"
        )

    def increase_sequence_idx(self, vis):
        self.vis_state.sequence_idx += 1
        if self.vis_state.sequence_idx >= len(self.vis_state.full_frame_matrix):
            self.vis_state.sequence_idx = 0
        self.draw_everything(vis, reset_view=False)

    def save_camera_pose(self, vis):
        camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
        save_path = self.screenshot_path / self.sequence_id / "camera.json"
        o3d.io.write_pinhole_camera_parameters(str(save_path), camera)
        print(f"Saved camera pose to {save_path}")

    def load_camera_pose(self, vis):
        camera = o3d.io.read_pinhole_camera_parameters(
            str(self.screenshot_path / self.sequence_id / "camera.json")
        )
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)
        print(f"Loaded camera pose from {self.screenshot_path / self.sequence_id / 'camera.json'}")

    def decrease_sequence_idx(self, vis):
        self.vis_state.sequence_idx -= 1
        if self.vis_state.sequence_idx < 0:
            self.vis_state.sequence_idx = len(self.vis_state.full_frame_matrix) - 1
        self.draw_everything(vis, reset_view=False)

    def increase_frame_idx(self, vis):
        self.vis_state.frame_idx += self.vis_state.frame_step_size
        if self.vis_state.frame_idx >= self.vis_state.full_frame_matrix.shape[1] - 1:
            self.vis_state.frame_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_frame_idx(self, vis):
        self.vis_state.frame_idx -= self.vis_state.frame_step_size
        if self.vis_state.frame_idx < 0:
            self.vis_state.frame_idx = self.vis_state.full_frame_matrix.shape[1] - 2
        self.draw_everything(vis, reset_view=False)

    def toggle_flow_lines(self, vis):
        self.vis_state.flow_color = self.vis_state.flow_color.cycle()
        self.draw_everything(vis, reset_view=False)

    def draw_everything(self, vis, reset_view=False):
        self.geometry_list.clear()
        print(
            f"Vis State: {self.get_current_method_name()}, frame {self.vis_state.frame_idx} - {self.vis_state.frame_idx + self.vis_state.full_frame_matrix.subsequence_length - 1}"
        )
        frame_list = self.vis_state.full_frame_matrix[
            self.vis_state.sequence_idx,
            self.vis_state.frame_idx,
        ]
        color_list = self._frame_list_to_color_list(len(frame_list))

        for idx, flow_frame in enumerate(frame_list):
            pc = flow_frame.pc.global_pc
            self.add_pointcloud(pc, color=color_list[idx])
            flowed_pc = flow_frame.pc.flow(flow_frame.flow).global_pc

            draw_color = self.vis_state.flow_color.rgb

            # Add flowed point cloud
            if (draw_color is not None) and (flowed_pc is not None) and (idx < len(frame_list) - 1):
                self.add_lineset(pc, flowed_pc, color=draw_color)
        vis.clear_geometries()
        self.render(vis, reset_view=reset_view)

    def _print_instructions(self):
        print("#############################################################")
        print("Flow moves from the gray point cloud to the white point cloud\n")
        print("Press F to toggle flow lines")
        print("Press left or right arrow to change starter_idx")
        print(f"Press S to save screenshot (saved to {self.screenshot_path.absolute()})")
        print("#############################################################")
