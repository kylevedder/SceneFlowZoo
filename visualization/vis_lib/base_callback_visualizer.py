from bucketed_scene_flow_eval.datastructures import O3DVisualizer
import open3d as o3d
from pathlib import Path
import datetime
import matplotlib


class BaseCallbackVisualizer(O3DVisualizer):

    def __init__(
        self,
        screenshot_path: Path = Path() / "screenshots",
        point_size: float = 0.1,
        line_width: float = 1.0,
        add_world_frame: bool = True,
    ):
        super().__init__(
            point_size=point_size, line_width=line_width, add_world_frame=add_world_frame
        )
        self.screenshot_path = screenshot_path
        self.initial_geometry_list = []

    def _get_screenshot_path(self) -> Path:
        return self.screenshot_path / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

    def _get_pose_path(self) -> Path:
        return self.screenshot_path / "camera.json"

    def save_screenshot(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        save_name = self._get_screenshot_path()
        save_name.parent.mkdir(exist_ok=True, parents=True)
        vis.capture_screen_image(str(save_name))

    def _frame_list_to_color_list(
        self, num_frames: int, name: str
    ) -> list[tuple[float, float, float]]:
        if name == "default":
            return self._frame_list_to_color_list_default(num_frames)
        elif name == "zebra":
            return self._frame_list_to_color_list_interleave(
                num_frames,
                (0, 0, 0),
                (0.5730392156862745, 0.6681371902573746, 0.6980391477451371),
            )

        return self._frame_list_to_color_list_matplotlib(num_frames, name)

    def _frame_list_to_color_list_matplotlib(
        self, num_frames: int, name: str
    ) -> list[tuple[float, float, float]]:
        cmap = matplotlib.cm.get_cmap(name)
        return [cmap(idx / (num_frames - 1))[:3] for idx in range(num_frames)]

    def _frame_list_to_color_list_interleave(
        self,
        num_frames: int,
        color1: tuple[float, float, float] = (0, 0, 0),
        color2: tuple[float, float, float] = (1, 1, 1),
    ) -> list[tuple[float, float, float]]:
        """
        Interleave between the two colors based on the number of frames in the list
        """

        def interleave_color(idx: int) -> tuple[float, float, float]:
            if idx % 2 == 0:
                return color1
            else:
                return color2

        return [interleave_color(idx) for idx in range(num_frames)]

    def _frame_list_to_color_list_default(
        self,
        num_frames: int,
        color1: tuple[float, float, float] = (0, 1, 0),
        color2: tuple[float, float, float] = (0, 0, 1),
    ) -> list[tuple[float, float, float]]:
        """
        Interpolate between the two colors based on the number of frames in the list
        """

        def interpolate_color(color1, color2, fraction):
            return tuple(c1 * (1 - fraction) + c2 * fraction for c1, c2 in zip(color1, color2))

        return [
            interpolate_color(color1, color2, idx / (num_frames - 1)) for idx in range(num_frames)
        ]

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        vis.register_key_callback(ord("S"), self.save_screenshot)

    def draw_everything(self, vis, reset_view=False):
        self.render(vis, reset_view=reset_view)

    def _print_instructions(self):
        print("#############################################################")
        print("Flow moves from the gray point cloud to the white point cloud\n")
        print(f"Press S to save screenshot (saved to {self.screenshot_path.absolute()})")
        print("#############################################################")

    def save_camera_pose(self, vis):
        camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
        save_path = self._get_pose_path()
        o3d.io.write_pinhole_camera_parameters(str(save_path), camera)
        print(f"Saved camera pose to {save_path}")

    def load_camera_pose(self, vis, camera_path: Path | None = None):
        if camera_path is None:
            camera_path = self._get_pose_path()
        assert camera_path.exists(), f"Camera path {camera_path} does not exist."
        camera = o3d.io.read_pinhole_camera_parameters(str(camera_path))
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)
        print(f"Loaded camera pose from {camera_path}")

    def _setup_point_size(self, vis):
        print("Setting point size to", self.point_size)
        vis.get_render_option().point_size = self.point_size

    def run(
        self,
        vis=o3d.visualization.VisualizerWithKeyCallback(),
        camera_pose_path: Path | None = None,
    ):
        self._print_instructions()
        vis.create_window()
        self._setup_point_size(vis)
        vis.get_render_option().background_color = (1, 1, 1)
        vis.get_view_control().set_up([0, 0, 1])
        self._register_callbacks(vis)
        self.initial_geometry_list = self.geometry_list.copy()
        self.draw_everything(vis, reset_view=True)
        if camera_pose_path is not None:
            self.load_camera_pose(vis, camera_pose_path)
        vis.run()
