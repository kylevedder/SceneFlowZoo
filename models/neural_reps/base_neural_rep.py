from abc import abstractmethod
from models import BaseModule
from dataloaders import BucketedSceneFlowInputSequence
from models.optimization.cost_functions import BaseCostProblem
from models.optimization.utils import EarlyStopping
from pytorch_lightning.loggers import Logger


class BaseNeuralRep(BaseModule):

    def optim_forward(
        self,
        batched_sequence: list[BucketedSceneFlowInputSequence],
        optim_step: int,
        early_stopping: EarlyStopping,
        logger: Logger,
    ) -> list[BaseCostProblem]:
        return [
            self.optim_forward_single(input_sequence, optim_step, early_stopping, logger)
            for input_sequence in batched_sequence
        ]

    @abstractmethod
    def optim_forward_single(
        self,
        input_sequence: BucketedSceneFlowInputSequence,
        optim_step: int,
        early_stopping: EarlyStopping,
        logger: Logger,
    ) -> BaseCostProblem:
        raise NotImplementedError()
