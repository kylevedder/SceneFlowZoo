from pathlib import Path
from bucketed_scene_flow_eval.utils import *
from paper_figures.plot_lib import *
import argparse

from paper_figures.figs_gigachad.fig_gigachad_by_ablation_depth import plot_depth_ablation_barchart
from paper_figures.figs_gigachad.fig_dynamic_norm_epe_bar import plot_dynamic_norm_epe_bar
from paper_figures.figs_gigachad.fig_per_metacatagory_bar import plot_per_metacatagory_bar

# Get path to methods from command line
parser = argparse.ArgumentParser()
parser.add_argument("output_folder", type=Path)
args = parser.parse_args()

save_folder = args.output_folder
save_folder.mkdir(exist_ok=True, parents=True)


#################

# Load data

av2_class_test_data_root_dir = Path("./paper_figures/perf_data/av2_test/bucketed_epe/")

av2_class_val_data_root_dir = Path("./paper_figures/perf_data/av2_val/bucketed_epe/")

# fmt: off
gigachad_test_ablations = [
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 4", "gigachad_4"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 6", "gigachad_6"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 8", "gigachad_8"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 8 + Fourtier", "gigachad_fourtier"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 10", "gigachad_10"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 12", "gigachad_12"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 14", "gigachad_14"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad Depth 16", "gigachad_16"),    
]

gigachad_val_ablations = [
    BucketedEvalStats(av2_class_val_data_root_dir, "Gigachad Depth 6", "gigachad_depth6"),
    BucketedEvalStats(av2_class_val_data_root_dir, "Gigachad Depth 8", "gigachad_depth8"),
    BucketedEvalStats(av2_class_val_data_root_dir, "Gigachad Depth 10", "gigachad_depth10"),
    BucketedEvalStats(av2_class_val_data_root_dir, "Gigachad Depth 12", "gigachad_depth12"),
    BucketedEvalStats(av2_class_val_data_root_dir, "Gigachad Depth 14", "gigachad_depth14"),
    BucketedEvalStats(av2_class_val_data_root_dir, "Gigachad Depth 16", "gigachad_depth16"),
]

class_bucketed_eval_stats = [
    BucketedEvalStats(av2_class_test_data_root_dir, "Gigachad (Ours)", "gigachad_10"),
    BucketedEvalStats(av2_class_test_data_root_dir, "Flow4D 5 Frame", "flow4d"),
    BucketedEvalStats(av2_class_test_data_root_dir, "TrackFlow", "trackflow"),
    BucketedEvalStats(av2_class_test_data_root_dir, "DeFlow++", "deflowpp"),
    BucketedEvalStats(av2_class_test_data_root_dir, "ICP Flow", "icp_flow"),
    BucketedEvalStats(av2_class_test_data_root_dir, "DeFlow", "deflow"),
    BucketedEvalStats(av2_class_test_data_root_dir, "SeFlow", "seflow"),
    BucketedEvalStats(av2_class_test_data_root_dir, "FastNSF", "fast_nsf"),
    BucketedEvalStats(av2_class_test_data_root_dir, "NSFP", "nsfp"),
    BucketedEvalStats(av2_class_test_data_root_dir, "FastFlow3D", "fastflow3d"),
    BucketedEvalStats(av2_class_test_data_root_dir, "ZeroFlow 1x", "zeroflow_1x"),
    BucketedEvalStats(av2_class_test_data_root_dir, "ZeroFlow 3x", "zeroflow_3x"),
    BucketedEvalStats(av2_class_test_data_root_dir, "ZeroFlow 5x", "zeroflow_5x"),
    BucketedEvalStats(av2_class_test_data_root_dir, "ZeroFlow XL 3x", "zeroflow_xl_3x"),
    BucketedEvalStats(av2_class_test_data_root_dir, "ZeroFlow XL 5x", "zeroflow_xl_5x"),
]

# fmt: on

# gigachad_ablations.sort(reverse=True)
class_bucketed_eval_stats.sort(reverse=True)

set_font(6)

av2_class_test_save_dir = save_folder / "av2_test"
av2_class_test_save_dir.mkdir(exist_ok=True, parents=True)

av2_class_val_save_dir = save_folder / "av2_val"
av2_class_val_save_dir.mkdir(exist_ok=True, parents=True)

plot_depth_ablation_barchart(gigachad_test_ablations, av2_class_test_save_dir)
plot_depth_ablation_barchart(gigachad_val_ablations, av2_class_val_save_dir)

plot_per_metacatagory_bar(gigachad_val_ablations, av2_class_val_save_dir)

plot_dynamic_norm_epe_bar(class_bucketed_eval_stats, av2_class_test_save_dir)
