from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from pointclouds import PointCloud, SE3, SE2
import numpy as np
from loader_utils import load_npz

from . import WaymoSupervisedFlowSequenceLoader, WaymoSupervisedFlowSequence


class WaymoUnsupervisedFlowSequence(WaymoSupervisedFlowSequence):
    pass


class WaymoUnsupervisedFlowSequenceLoader(WaymoSupervisedFlowSequenceLoader):

    def load_sequence(self, log_id: str) -> WaymoSupervisedFlowSequence:
        sequence_folder = self.log_lookup[log_id]
        return WaymoUnsupervisedFlowSequence(sequence_folder,
                                             verbose=self.verbose)
