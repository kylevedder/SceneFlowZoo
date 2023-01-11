from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SequenceDataset

from nsfp_baseline import NSFPProcessor, NSFPLoss

from tqdm import tqdm

import torch

SEQUENCE_LENGTH = 6
BATCH_SIZE = 1


def main_fn():
    sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/test/')
    dataset = SequenceDataset(sequence_loader, SEQUENCE_LENGTH)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    nsfp_processor = NSFPProcessor()
    loss_fn = NSFPLoss()

    for batch_idx, subsequence_batch in enumerate(dataloader):
        pc_array = subsequence_batch["pc_array_stack"][0].float()
        loss = 0
        for subsequence_idx in tqdm(range(SEQUENCE_LENGTH - 1)):
            pc1 = pc_array[subsequence_idx]
            pc2 = pc_array[subsequence_idx + 1]
            pc1 = pc1[torch.isfinite(pc1[:, 0])]
            pc2 = pc2[torch.isfinite(pc2[:, 0])]
            pc1 = torch.unsqueeze(pc1, 0).cuda()
            pc2 = torch.unsqueeze(pc2, 0).cuda()
            warped_pc, target_pc = nsfp_processor(pc1, pc2)
            loss += loss_fn(warped_pc, target_pc)
        print("Batch: ", batch_idx, " Loss: ", loss)
        if batch_idx > 3:
            break


if __name__ == "__main__":
    main_fn()