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

To generate the flying bird plot:

```bash
python visualization/visualize_flow_feathers.py --sequence_id 5f016e44-0f38-3837-9111-58ec18d1a5e6 --sequence_folder /efs/argoverse2/test --point_size 0.1 --sequence_length 150 --frame_idx_start 0 --flow_folders /efs/argoverse2/test_sceneflow_feather /efs/argoverse2/test_bucketed_nsfp_distillation_5x_out/ /efs/argoverse2/test_gigachad_nsf_depth18_flattened/ /efs/argoverse2/test_fast_nsf_flow_feather/sequence_len_002 --camera_pose_file paper_figures/figs_gigachad/3d_vis_camera_configs/flying_bird.json
```

Press up and down arrows to cycle through the different flow folders.

