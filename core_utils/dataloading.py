import torch
from typing import Any
import dataloaders
from dataloaders import EvalWrapper


def make_dataloader(
    dataset_name: str, dataset_args: dict[str, Any], dataloader_args: dict[str, Any]
) -> tuple[torch.utils.data.DataLoader, EvalWrapper]:
    dataset = dataloaders.construct_dataset(dataset_name, dataset_args)
    dataloader = torch.utils.data.DataLoader(
        dataset, **dataloader_args, collate_fn=dataset.collate_fn
    )
    return dataloader, dataset.evaluator()
