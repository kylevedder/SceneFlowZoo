import torch
from models import BaseModel, ForwardMode
from typing import Any, List
from dataloaders import (
    BucketedSceneFlowInputSequence,
    BucketedSceneFlowOutputSequence,
)
from pytorch_lightning.loggers import Logger

class Flow4D(BaseModel):
    def __init__(self, cfg: Any, evaluator: Any = None):
        super().__init__()
        self.cfg = cfg

        self.loss_fn = self.initialize_loss_fn(cfg)

    def forward(
        self,
        forward_mode: ForwardMode,
        batched_sequence: List[BucketedSceneFlowInputSequence],
        logger: Logger,
    ) -> List[BucketedSceneFlowOutputSequence]:
        """
        Args:
            batched_sequence: A list (len=batch size) of BucketedSceneFlowItems.

        Returns:
            A list (len=batch size) of BucketedSceneFlowOutputItem.
        """
        if forward_mode == ForwardMode.TRAIN:
            return self.train_forward(batched_sequence, logger)
        elif forward_mode == ForwardMode.VAL:
            return self.val_forward(batched_sequence, logger)
        else:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")

    def loss_fn(
        self, 
        input_batch: List[BucketedSceneFlowInputSequence],
        model_res: List[BucketedSceneFlowOutputSequence],
    ) -> dict[str, torch.Tensor]:

        return self.loss_fn(input_batch, model_res)

    def _model_forward(
        self, batch, batch_idx
    ):
        res_dict = self.model(batch)

        # compute loss
        total_loss = 0.0
        total_flow_loss = 0.0

        batch_sizes = len(batch["pose0"])
        gt_flow = batch['flow'] #gt_flow = ego+motion

        pose_flows = res_dict['pose_flow'] #pose_flow = ego-motion's flow
        pc0_valid_idx = res_dict['pc0_valid_point_idxes'] # since padding
        est_flow = res_dict['flow'] #network's output, motion flow 

        
        for batch_id in range(batch_sizes):
            pc0_valid_from_pc2res = pc0_valid_idx[batch_id]
            pose_flow_ = pose_flows[batch_id][pc0_valid_from_pc2res]
            est_flow_ = est_flow[batch_id]
            gt_flow_ = gt_flow[batch_id][pc0_valid_from_pc2res]
            gt_flow_ = gt_flow_ - pose_flow_ 

            res_dict = {'est_flow': est_flow_, 
                        'gt_flow': gt_flow_, 
                        'gt_classes': None if 'flow_category_indices' not in batch else batch['flow_category_indices'][batch_id][pc0_valid_from_pc2res], #CLASS 0~30
                        }
                    
            loss = self.loss_fn(res_dict)
            total_flow_loss += loss.item()

            total_loss += loss
        
        self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size)

        return total_loss
