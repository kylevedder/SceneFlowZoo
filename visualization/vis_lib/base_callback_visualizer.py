from bucketed_scene_flow_eval.datastructures import O3DVisualizer
import open3d as o3d
from pathlib import Path
import datetime


class BaseCallbackVisualizer(O3DVisualizer):

    def __init__(self, screenshot_path: Path = Path() / "screenshots", point_size: float = 0.1):
        super().__init__(point_size=point_size)
        self.screenshot_path = screenshot_path

    def _get_screenshot_path(self) -> Path:
        return self.screenshot_path / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

    def save_screenshot(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        save_name = self._get_screenshot_path()
        save_name.parent.mkdir(exist_ok=True, parents=True)
        vis.capture_screen_image(str(save_name))

    def _frame_list_to_color_list(
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

    def run(self):
        self._print_instructions()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.get_render_option().background_color = (1, 1, 1)
        vis.get_view_control().set_up([0, 0, 1])
        self._register_callbacks(vis)
        self.draw_everything(vis, reset_view=True)
        vis.run()
