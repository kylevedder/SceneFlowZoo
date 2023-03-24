import numpy as np
from loader_utils import load_npz
from pathlib import Path
from typing import List, Tuple, Dict, Any

from dataloaders import WaymoSupervisedFlowSequenceLoader


root_path = Path('/efs/waymo_open_preprocessed/train')
sequence_loader = WaymoSupervisedFlowSequenceLoader(root_path)

for sequence_id in sequence_loader.get_sequence_ids():
    sequence = sequence_loader.load_sequence(sequence_id)
    for seq_idx in range(len(sequence)):
        frame = sequence.load(seq_idx, seq_idx)
        print(frame)
    


# for (before_frame, after_frame) in look_up_table:
#     before_frame = Frame(*before_frame)
#     after_frame = Frame(*after_frame)
#     breakpoint()