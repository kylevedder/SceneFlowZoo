from abc import abstractmethod
from models import BaseModel
from dataloaders import BucketedSceneFlowInputSequence
from models.optimization.cost_functions import BaseCostProblem


class BaseNeuralRep(BaseModel):

    def optim_forward(
        self, batched_sequence: list[BucketedSceneFlowInputSequence]
    ) -> list[BaseCostProblem]:
        return [self.optim_forward_single(input_sequence) for input_sequence in batched_sequence]

    @abstractmethod
    def optim_forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BaseCostProblem:
        raise NotImplementedError()
