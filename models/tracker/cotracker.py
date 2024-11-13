from .base_tracker import BasePointTracker3D, ImageSpaceTracks, VideoInfo, VideoDirection

from cotracker.predictor import CoTrackerPredictor
from pathlib import Path


class CoTracker3D(BasePointTracker3D):
    def __init__(
        self,
        camera_infos: list[tuple[str, VideoDirection | str]],
        cotracker_checkpoint: Path = Path(
            "/payload_files/cache/torch/hub/checkpoints/cotracker2.pth"
        ),
    ):
        super().__init__(camera_infos)
        assert (
            cotracker_checkpoint.exists()
        ), f"CoTracker checkpoint not found at {cotracker_checkpoint}"
        self.tracker = CoTrackerPredictor(
            checkpoint=cotracker_checkpoint,
        )

    def _compute_video_tracks(self, video_info: VideoInfo) -> ImageSpaceTracks:
        video_info = video_info.to(self.device)
        pred_tracks, pred_visibility = self.tracker(
            video=video_info.video_tensor, queries=video_info.get_query_points()
        )
        return ImageSpaceTracks(tracks=pred_tracks, visibility=pred_visibility)
