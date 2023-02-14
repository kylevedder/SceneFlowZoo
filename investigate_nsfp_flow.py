from pathlib import Path
import numpy as np

from dataloaders import ArgoverseRawSequenceLoader

argoverse_data = Path("/efs/argoverse2")

train_sequence_dir = argoverse_data / "train"
flow_sequence_dir = argoverse_data / "train_nsfp_flow"

flow_sequence_list = sorted(flow_sequence_dir.glob("*/"))
print(f"Found {len(flow_sequence_list)} flow sequences")
# check if all sequences are the same
flow_lookup = {x.stem: x for x in flow_sequence_list}
flow_names_list = sorted(flow_lookup.keys())


def load_npz(path):
    return dict(np.load(path))


flow_sequence_entries = [
    (name, sorted(flow_lookup[name].glob("*.npz"))) for name in flow_names_list
]

seq_loader = ArgoverseRawSequenceLoader(argoverse_data / "train", log_subset=flow_names_list)

for name, flow_sequence in flow_sequence_entries:
    for flow in flow_sequence:
        npz = load_npz(flow)

        breakpoint()



# names_set = train_names_set.intersection(flow_names_set)

# train_sequence_entries = [
#     sorted(train_lookup[name].glob("*.npz")) for name in names_set
# ]

# for flow_sequence in flow_sequence_entries:

#     for entry in flow_sequence:
#         npz = load_npz(entry)
#         breakpoint()
