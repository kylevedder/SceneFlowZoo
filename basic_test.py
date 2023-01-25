import torch
import torch.nn as nn

from accelerate import Accelerator
import numpy as np
from tqdm import tqdm

accelerator = Accelerator()


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x):
        return {'res': self.fc(x)}


class MyDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.data = [np.ones((6, ), dtype=np.float32) * i for i in range(1000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = MyDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=16)

torch.random.manual_seed(0)
model = MyModule()

res_lst = []

model, dataloader = accelerator.prepare(model, dataloader)

model.eval()
for batch_idx, batch in enumerate(tqdm(dataloader)):
    accelerator.print(f"Batch {batch_idx}")
    with torch.no_grad():
        res = model(batch)
    gathered_res = accelerator.gather_for_metrics(res)
    res_lst.append(gathered_res)

accelerator.print(sum([e['res'].sum() for e in res_lst]))
