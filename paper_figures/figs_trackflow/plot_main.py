from pathlib import Path
from bucketed_scene_flow_eval.utils import *
from paper_figures.plot_lib import *
import argparse
from paper_figures.figs_trackflow.fig_dynamic_norm_epe_bar import plot_dynamic_norm_epe_bar
from paper_figures.figs_trackflow.fig_threeway_epe_bar import (
    plot_threeway_epe_bar,
    plot_threeway_dynamic_epe_bar,
)
from paper_figures.figs_trackflow.fig_per_metacatagory_bar import plot_per_metacatagory_bar


# Get path to methods from command line
parser = argparse.ArgumentParser()
parser.add_argument("output_folder", type=Path)
args = parser.parse_args()

save_folder = args.output_folder
save_folder.mkdir(exist_ok=True, parents=True)


#################

# Load data

av2_class_test_data_root_dir = Path("./paper_figures/perf_data/av2_test/")
av2_class_test_bucketed_data_root_dir = av2_class_test_data_root_dir / "bucketed_epe"
av2_class_test_threeway_data_root_dir = av2_class_test_data_root_dir / "threeway_epe"

av2_volume_test_data_root_dir = Path("./paper_figures/perf_data/av2_test_volume_split/")
av2_volume_test_bucketed_data_root_dir = av2_volume_test_data_root_dir / "bucketed_epe"


# fmt: off

class_bucketed_eval_stats = [
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "Flow4D 5 Frame", "flow4d"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "TrackFlow", "trackflow"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "DeFlow++", "deflowpp"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "ICP Flow", "icp_flow"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "DeFlow", "deflow"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "Liu et al. 2024", "lie_et_al"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "SeFlow", "seflow"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "FastNSF", "fast_nsf"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "NSFP", "nsfp"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "FastFlow3D", "fastflow3d"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "ZeroFlow 1x", "zeroflow_1x"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "ZeroFlow 3x", "zeroflow_3x"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "ZeroFlow 5x", "zeroflow_5x"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "ZeroFlow XL 3x", "zeroflow_xl_3x"),
    BucketedEvalStats(av2_class_test_bucketed_data_root_dir, "ZeroFlow XL 5x", "zeroflow_xl_5x"),
]

class_threeway_eval_stats = [
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "TrackFlow (ours)", "test_Tracktor_LE3DE2E_c02_clamped_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "DeFlow", "test_DeFlow_submission_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "NSFP", "test_nsfp_flow_feather_av2_2024_sf_submission_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "FastFlow3D", "test_bucketed_supervised_out_av2_2024_sf_submission_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "ZeroFlow 1x", "test_bucketed_nsfp_distillation_1x_out_av2_2024_sf_submission_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "ZeroFlow 3x", "test_bucketed_nsfp_distillation_3x_out_av2_2024_sf_submission_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "ZeroFlow 5x", "test_bucketed_nsfp_distillation_5x_out_av2_2024_sf_submission_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "ZeroFlow XL 3x", "test_bucketed_nsfp_distillation_xl_3x_out_av2_2024_sf_submission_per_class_results_35"),
    ThreewayEvalStats(av2_class_test_threeway_data_root_dir, "ZeroFlow XL 5x", "test_bucketed_nsfp_distillation_xl_5x_out_av2_2024_sf_submission_per_class_results_35"),
]

volume_bucketed_eval_stats = [
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "TrackFlow (ours)", "trackflow"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "DeFlow", "deflow"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "NSFP", "nsfp"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "FastFlow3D", "fastflow3d"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "ZeroFlow 1x", "zeroflow_1x"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "ZeroFlow 3x", "zeroflow_3x"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "ZeroFlow 5x", "zeroflow_5x"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "ZeroFlow XL 3x", "zeroflow_xl_3x"),
    BucketedEvalStats(av2_volume_test_bucketed_data_root_dir, "ZeroFlow XL 5x", "zeroflow_xl_5x"),
]

# fmt: on


class_bucketed_eval_stats.sort(reverse=True)
class_threeway_eval_stats.sort(reverse=True)
volume_bucketed_eval_stats.sort(reverse=True)

set_font(6)

av2_class_test_save_dir = save_folder / "av2_test"
av2_class_test_save_dir.mkdir(exist_ok=True, parents=True)

plot_per_metacatagory_bar(class_bucketed_eval_stats, av2_class_test_save_dir)
plot_dynamic_norm_epe_bar(class_bucketed_eval_stats, av2_class_test_save_dir)

# assert len(class_bucketed_eval_stats) == len(
#     class_threeway_eval_stats
# ), f"Lengths do not match: {len(class_bucketed_eval_stats)} != {len(class_threeway_eval_stats)}"

# joint_eval_stats = list(zip(class_bucketed_eval_stats, class_threeway_eval_stats))

# # Verify that each tuple has the same name
# for bucketed, threeway in joint_eval_stats:
#     assert bucketed.name == threeway.name, f"Names do not match: {bucketed.name} != {threeway.name}"

# plot_threeway_epe_bar(joint_eval_stats, av2_class_test_save_dir)
# plot_threeway_dynamic_epe_bar(joint_eval_stats, av2_class_test_save_dir)

av2_volume_test_save_dir = save_folder / "av2_volume_test"
av2_volume_test_save_dir.mkdir(exist_ok=True, parents=True)

plot_per_metacatagory_bar(volume_bucketed_eval_stats, av2_volume_test_save_dir)
plot_dynamic_norm_epe_bar(volume_bucketed_eval_stats, av2_volume_test_save_dir)
