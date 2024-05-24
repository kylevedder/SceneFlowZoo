import torch
from pytorch3d.ops import knn_points
from bucketed_scene_flow_eval.datastructures import (
    TimeSyncedSceneFlowFrame,
    PointCloud,
    PointCloudFrame,
    RGBFrame,
    CameraProjection,
    O3DVisualizer,
)
from pytorch_lightning.loggers.logger import Logger
from dataloaders import RawFullFrameInputSequence, RawFullFrameOutputSequence
from models import BaseRawModel
import numpy as np

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

from dataclasses import dataclass


@dataclass
class ProjectedPoints:
    image_space_points: torch.Tensor
    depths: torch.Tensor
    camera_projection: CameraProjection
    rgb_frame: RGBFrame

    def __post_init__(self):
        # Type checks
        assert isinstance(
            self.image_space_points, torch.Tensor
        ), f"Expected image space points to be a tensor, got {type(self.image_space_points)}"
        assert isinstance(
            self.depths, torch.Tensor
        ), f"Expected depths to be a tensor, got {type(self.depths)}"
        assert isinstance(
            self.camera_projection, CameraProjection
        ), f"Expected camera projection to be a CameraProjection object, got {type(self.camera_projection)}"
        assert isinstance(
            self.rgb_frame, RGBFrame
        ), f"Expected RGB frame to be a RGBFrame object, got {type(self.rgb_frame)}"

        assert (
            self.image_space_points.dim() == 2
        ), f"Expected 2D image space points tensor, got {self.image_space_points.dim()}"
        assert self.depths.dim() == 1, f"Expected 1D depths tensor, got {self.depths.dim()}"
        assert (
            self.image_space_points.shape[0] == self.depths.shape[0]
        ), f"Expected same number of points in image space and depths, got {self.image_space_points.shape[0]} and {self.depths.shape[0]}"
        assert (
            self.image_space_points.dtype == torch.int
        ), f"Expected image space points to be of type int, got {self.image_space_points.dtype}"
        assert (
            self.depths.dtype == torch.float
        ), f"Expected depths to be of type float, got {self.depths.dtype}"

    def to(self, device: torch.device) -> "ProjectedPoints":
        self = ProjectedPoints(
            image_space_points=self.image_space_points.to(device),
            depths=self.depths.to(device),
            camera_projection=self.camera_projection,
            rgb_frame=self.rgb_frame,
        )
        return self


@dataclass
class VideoInfo:
    video_tensor: torch.Tensor
    projected_points: list[ProjectedPoints]

    def __post_init__(self):
        assert (
            self.video_tensor.dim() == 5
        ), f"Expected 5D video tensor, got {self.video_tensor.dim()}"
        assert (
            self.video_tensor.shape[0] == 1
        ), f"Expected batch size of 1, got {self.video_tensor.shape[0]}"
        assert (
            self.video_tensor.dtype == torch.float
        ), f"Expected video tensor to be of type float, got {self.video_tensor.dtype}"

        assert self.video_tensor.shape[1] == len(self.projected_points), (
            f"Expected video tensor to have the same number of frames as projected points, got "
            f"{self.video_tensor.shape[1]} and {len(self.projected_points)}"
        )

    def get_query_points(self) -> torch.Tensor:
        query_projected_points = self.projected_points[0]
        pixels = query_projected_points.image_space_points
        # Add an extra dimension to create a batch dimension
        return torch.cat([torch.zeros_like(pixels[:, :1]), pixels], dim=1)[None].float()

    def to(self, device: torch.device) -> "VideoInfo":
        self = VideoInfo(
            video_tensor=self.video_tensor.to(device),
            projected_points=[
                projected_points.to(device) for projected_points in self.projected_points
            ],
        )
        return self


@dataclass
class ImageSpaceTracks:
    tracks: torch.Tensor
    visibility: torch.Tensor

    def __len__(self):
        return self.tracks.shape[1]

    def __getitem__(self, idx):
        return self.tracks[0, idx], self.visibility[0, idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __post_init__(self):
        assert self.tracks.dim() == 4, f"Expected 4D tracks tensor, got {self.tracks.dim()}"
        assert self.tracks.shape[0] == 1, f"Expected batch size of 1, got {self.tracks.shape[0]}"
        assert (
            self.visibility.dim() == 3
        ), f"Expected 3D visibility tensor, got {self.visibility.dim()}"
        assert self.tracks.shape[:2] == self.visibility.shape[:2], (
            f"Expected tracks and visibility to have the same number of frames, got "
            f"{self.tracks.shape[:2]} and {self.visibility.shape[:2]}"
        )
        assert self.tracks.shape[2] == self.visibility.shape[2], (
            f"Expected tracks and visibility to have the same number of points, got "
            f"{self.tracks.shape[2]} and {self.visibility.shape[2]}"
        )

    def to(self, device: torch.device) -> "ImageSpaceTracks":
        self = ImageSpaceTracks(
            tracks=self.tracks.to(device),
            visibility=self.visibility.to(device),
        )
        return self


class PointTracker3D(BaseRawModel):

    def __init__(self, camera_names: list[str]) -> None:
        super().__init__()
        self.cotracker = CoTrackerPredictor(
            checkpoint="/payload_files/cache/torch/hub/checkpoints/cotracker2.pth"
        )
        assert len(camera_names) > 0, f"Must have at least one camera name"
        self.camera_names = camera_names

    @property
    def device(self):
        return next(self.parameters()).device

    def _project_points(self, pc_frame: PointCloudFrame, rgb_frame: RGBFrame) -> ProjectedPoints:
        full_rgb_image = rgb_frame.rgb
        rgb_image = full_rgb_image.masked_image
        camera_projection = rgb_frame.camera_projection.to_masked_projection(full_rgb_image)

        pc_into_cam_frame_se3 = pc_frame.pose.sensor_to_ego.compose(
            rgb_frame.pose.sensor_to_ego.inverse()
        )
        cam_frame_pc = pc_frame.pc.transform(pc_into_cam_frame_se3)
        # Filter out all the points behind the camera.
        cam_frame_pc = PointCloud(cam_frame_pc.points[cam_frame_pc.points[:, 0] > 0])

        projected_points = camera_projection.camera_frame_to_pixels(cam_frame_pc.points).astype(
            np.int32
        )
        projected_points_mask = (
            (projected_points[:, 0] >= 0)
            & (projected_points[:, 0] < rgb_image.shape[1])
            & (projected_points[:, 1] >= 0)
            & (projected_points[:, 1] < rgb_image.shape[0])
        )

        valid_projected_points = projected_points[projected_points_mask]
        # X is the distance away from the camera in the right hand coordinate frame.
        valid_depths = cam_frame_pc.points[projected_points_mask, 0].astype(np.float32)
        return ProjectedPoints(
            image_space_points=torch.from_numpy(valid_projected_points),
            depths=torch.from_numpy(valid_depths),
            camera_projection=camera_projection,
            rgb_frame=rgb_frame,
        )

    def _extract_video_info(
        self, camera_name: str, frame_list: list[TimeSyncedSceneFlowFrame]
    ) -> VideoInfo:
        rgb_frames = [frame.rgbs[camera_name] for frame in frame_list]
        np_frames = np.array([rgb.rgb.masked_image.full_image for rgb in rgb_frames])
        video_tensor = torch.tensor(np_frames).permute(0, 3, 1, 2)[None].float() * 255.0
        projected_points = [
            self._project_points(frame.pc, rgb_frame)
            for frame, rgb_frame in zip(frame_list, rgb_frames)
        ]

        return VideoInfo(
            video_tensor=video_tensor,
            projected_points=projected_points,
        )

    def _compute_video_tracks(self, video_info: VideoInfo) -> ImageSpaceTracks:
        video_info = video_info.to(self.device)
        pred_tracks, pred_visibility = self.cotracker(
            video=video_info.video_tensor, queries=video_info.get_query_points()
        )
        return ImageSpaceTracks(tracks=pred_tracks, visibility=pred_visibility)

    def _knn2d(self, target_pixels: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        Given the Kx2 target pixels, and Nx2 query points, return the indices of the nearest point for all N query points.

        All points are in 2D space.
        """
        # Add a batch dimension to both target_pixels and query_points
        target_pixels = target_pixels.unsqueeze(0)  # Shape: (1, K, 2)
        query_points = query_points.unsqueeze(0)  # Shape: (1, N, 2)

        # Perform KNN search
        knn_result = knn_points(query_points, target_pixels, K=1)

        # Extract the indices of the nearest neighbors
        nearest_indices = knn_result.idx.squeeze(0).squeeze(-1)  # Shape: (N,)
        assert nearest_indices.shape == query_points.shape[1:2], (
            f"Expected the same number of nearest indices as query points, got "
            f"{nearest_indices.shape} and {query_points.shape}"
        )
        return nearest_indices

    def _image_track_to_3d(
        self, projected_points: list[ProjectedPoints], image_track: ImageSpaceTracks
    ) -> list[PointCloud]:
        assert len(projected_points) == len(image_track), (
            f"Expected the same number of projected points and image tracks, got "
            f"{len(projected_points)} and {len(image_track)}"
        )

        global_pc_list: list[PointCloud] = []
        for idx, (projected_point, (track, is_visible)) in enumerate(
            zip(projected_points, image_track)
        ):

            projected_point = projected_point.to(track.device)
            target_pixels = projected_point.image_space_points.float()

            if idx == 0:
                assert torch.all(
                    is_visible
                ), f"Expected all points to be visible in the first frame"

            # Handle KNN on the visible tracks
            visible_tracks = track[is_visible]
            nearest_projected_point_idxes = self._knn2d(target_pixels, visible_tracks)

            visible_pixel_coordinates = projected_point.image_space_points[
                nearest_projected_point_idxes
            ]
            visible_pixel_coordinate_depths = projected_point.depths[
                nearest_projected_point_idxes
            ].unsqueeze(1)

            visible_pixel_coordinates = visible_pixel_coordinates.cpu().numpy()
            visible_pixel_coordinate_depths = visible_pixel_coordinate_depths.cpu().numpy()

            visible_camera_frame_points = projected_point.camera_projection.to_camera(
                visible_pixel_coordinates, visible_pixel_coordinate_depths
            )

            visible_camera_frame_pc = PointCloud(visible_camera_frame_points)

            visible_global_pc = visible_camera_frame_pc.transform(
                projected_point.rgb_frame.pose.sensor_to_global
            )

            full_camera_frame_points = np.zeros((len(track), 3))
            is_visible_np = is_visible.cpu().numpy()
            full_camera_frame_points[is_visible_np] = visible_global_pc.points

            if idx == 0:
                assert np.all(
                    is_visible_np
                ), f"Expected all points to be visible in the first frame"

            # Create a full point cloud with all the points
            if idx > 0:
                # First frame should not have any invisible points
                if idx > 0:
                    # Set the invisible points to NaN
                    # These will need to be cleaned up later
                    full_camera_frame_points[~is_visible_np] = global_pc_list[-1].points[
                        ~is_visible_np
                    ]

            global_pc_list.append(PointCloud(full_camera_frame_points))

        return global_pc_list

    def _visualize_rgb_track(
        self, camera_name: str, video_info: VideoInfo, image_track: ImageSpaceTracks
    ):
        rgb_vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        print(f"Visualizing image track for camera {camera_name}")
        rgb_vis.visualize(
            video=video_info.video_tensor,
            tracks=image_track.tracks,
            visibility=image_track.visibility,
            filename=f"video_{camera_name}",
        )

    def inference_forward_single(
        self,
        input_sequence: RawFullFrameInputSequence,
        logger: Logger,
    ) -> RawFullFrameOutputSequence:
        frame_list = input_sequence.frame_list

        o3d_vis = O3DVisualizer(point_size=3)

        for frame in frame_list:
            o3d_vis.add_global_pc_frame(frame.pc, color=[0, 0, 1])

        for camera_name in self.camera_names:
            print(f"Processing camera {camera_name}")
            video_info = self._extract_video_info(camera_name, frame_list)
            print(f"Running cotracker")
            image_track = self._compute_video_tracks(video_info)

            self._visualize_rgb_track(camera_name, video_info, image_track)

            print(f"Converting image track to 3D")
            projected_pc_frames = self._image_track_to_3d(video_info.projected_points, image_track)

            for pc in projected_pc_frames:
                # Add the raw point clouds
                o3d_vis.add_pointcloud(pc, color=[1, 0, 0])

        o3d_vis.run()

        breakpoint()

        return None

    def loss_fn(
        self,
        input_batch: list[RawFullFrameInputSequence],
        model_res: list[RawFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()