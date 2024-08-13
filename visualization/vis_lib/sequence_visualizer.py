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
    sequence_idx: int  # Index into the list of possible sequences (different methods)
    subsequence_length_idx: int  # Index into teh subsequence length (different frame lengths)
    frame_idx: int  # Index into the frame list (only relevant if the subsequence is not the length of the full sequence)
    frame_step_size: int
    flow_color: ColorEnum = ColorEnum.RED

    def __str__(self):
        return f"sequence_idx: {self.sequence_idx}, frame__idx: {self.frame_idx}, "


class SequenceVisualizer(BaseCallbackVisualizer):

    def __init__(
        self,
        name_sequence_list: list[tuple[str, list[TimeSyncedSceneFlowFrame]]],
        sequence_id: str,
        frame_idx: int = 0,
        subsequence_lengths: list[int] = [2],
        point_size: float = 0.1,
        step_size: int = 1,
        color_map_name: str = "default",
    ):
        super().__init__(point_size=point_size)
        self.sequence_id = sequence_id
        self.name_lst = [name for name, _ in name_sequence_list]
        self.sequence_lst = [sequence for _, sequence in name_sequence_list]
        self.subsequence_lengths = subsequence_lengths
        self.vis_state = VisState(
            frame_idx=frame_idx,
            frame_step_size=step_size,
            sequence_idx=0,
            subsequence_length_idx=0,
        )
        self.color_map_name = color_map_name

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        vis.register_key_callback(ord("S"), self.save_screenshot)
        vis.register_key_callback(ord("F"), self.toggle_flow_lines)
        vis.register_key_callback(ord("C"), self.save_camera_pose)
        vis.register_key_callback(ord("V"), self.load_camera_pose)
        vis.register_key_callback(ord("L"), self.increase_subsequence_length_idx)
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
            / f"{self.vis_state.frame_idx:06d}_{self.get_current_method_name()}_subsequence_length_{self.get_current_subsequence_length()}.png"
        )

    def get_num_sequences(self) -> int:
        return len(self.sequence_lst)

    def get_current_subsequence_length(self) -> int:
        return self.subsequence_lengths[self.vis_state.subsequence_length_idx]

    def get_current_full_sequence(self) -> list[TimeSyncedSceneFlowFrame]:
        return self.sequence_lst[self.vis_state.sequence_idx]

    def get_current_subsequence(self) -> list[TimeSyncedSceneFlowFrame]:
        full_sequence = self.get_current_full_sequence()
        subsequence_length = self.get_current_subsequence_length()
        return full_sequence[
            self.vis_state.frame_idx : self.vis_state.frame_idx + subsequence_length
        ]

    def increase_sequence_idx(self, vis):
        self.vis_state.sequence_idx += 1
        if self.vis_state.sequence_idx >= len(self.get_current_full_sequence()):
            self.vis_state.sequence_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_sequence_idx(self, vis):
        self.vis_state.sequence_idx -= 1
        if self.vis_state.sequence_idx < 0:
            self.vis_state.sequence_idx = len(self.get_current_full_sequence()) - 1
        self.draw_everything(vis, reset_view=False)

    def increase_subsequence_length_idx(self, vis):
        self.vis_state.subsequence_length_idx += 1
        if self.vis_state.subsequence_length_idx >= len(self.subsequence_lengths):
            self.vis_state.subsequence_length_idx = 0
        self.draw_everything(vis, reset_view=False)

    def increase_frame_idx(self, vis):
        self.vis_state.frame_idx += self.vis_state.frame_step_size
        if self.vis_state.frame_idx >= len(self.get_current_full_sequence()) - 1:
            self.vis_state.frame_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_frame_idx(self, vis):
        self.vis_state.frame_idx -= self.vis_state.frame_step_size
        if self.vis_state.frame_idx < 0:
            self.vis_state.frame_idx = len(self.get_current_full_sequence()) - 2
        self.draw_everything(vis, reset_view=False)

    def toggle_flow_lines(self, vis):
        self.vis_state.flow_color = self.vis_state.flow_color.cycle()
        self.draw_everything(vis, reset_view=False)

    def draw_everything(self, vis, reset_view=False):
        self.geometry_list.clear()
        print(
            f"Vis State: {self.get_current_method_name()}, frame {self.vis_state.frame_idx} - {self.vis_state.frame_idx + self.get_current_subsequence_length() - 1}"
        )
        frame_list = self.get_current_subsequence()
        color_list = self._frame_list_to_color_list(len(frame_list), self.color_map_name)

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
        print("Press C to save camera pose")
        print("Press V to load camera pose")
        print("Press L to change subsequence length idx")
        print("Press left or right arrow to change starter_idx")
        print(f"Press S to save screenshot (saved to {self.screenshot_path.absolute()})")
        print("#############################################################")
