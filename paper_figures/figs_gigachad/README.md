# How to generate all paper figures

All commands are from the root of the repo, unless otherwise specified.

To generate all 2D matplotlib plots:

```bash
python3 paper_figures/figs_gigachad/plot_main.py
```

To generate the first 3D pedestrian plot:

```bash
python visualization/visualize_flow_feathers.py --sequence_id 04994d08-156c-3018-9717-ba0e29be8153 --sequence_folder /efs/argoverse2/val --flow_folders /efs/argoverse2/val_sceneflow_feather  --point_size 0.1 --sequence_length 150 --frame_idx_start 85 --camera_pose_file paper_figures/figs_gigachad/3d_vis_camera_configs/pedestrian_teaser.json
```

Press F to toggle the flow, and press L to change the length of the subsequence.

