import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class EnhancedSessionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        seq_items = row['items'][:-1]
        seq_cats = row['cats'][:-1]
        seq_evts = row['events'][:-1]

        target = row['items'][-1]

        # Oturum Grafiği İnşası
        unique_nodes, inverse = np.unique(seq_items, return_inverse=True)
        adj = np.zeros((len(unique_nodes), len(unique_nodes)))
        for i in range(len(inverse)-1):
            adj[inverse[i], inverse[i+1]] = 1

        last_idx_local = inverse[-1]

        return (
            torch.tensor(adj, dtype=torch.float),
            torch.tensor(unique_nodes, dtype=torch.long),
            torch.tensor(seq_cats, dtype=torch.long),
            torch.tensor(seq_evts, dtype=torch.long),
            torch.tensor(last_idx_local, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )

def collate_fn(batch, device):
    adjs, nodes, cats, evts, lasts, targets = zip(*batch)
    max_nodes = max(len(n) for n in nodes)

    pad_adjs, pad_nodes, pad_cats, pad_evts, masks = [], [], [], [], []

    for i in range(len(batch)):
        n = adjs[i].shape[0]
        pad_adj = F.pad(adjs[i], (0, max_nodes-n, 0, max_nodes-n))
        pad_adjs.append(pad_adj)

        pad_node = F.pad(nodes[i], (0, max_nodes-n))
        pad_nodes.append(pad_node)

        # Gerçek düğümleri belirten maskeleme
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)

        # Global bağlam için son tıklanan nesnenin kategorisini ve olay türünü alma
        pad_cats.append(cats[i][-1])
        pad_evts.append(evts[i][-1])

    return (
        torch.stack(pad_adjs).to(device),
        torch.stack(pad_nodes).to(device),
        torch.tensor(pad_cats).to(device),
        torch.tensor(pad_evts).to(device),
        torch.tensor(lasts).to(device),
        torch.stack(masks).to(device),
        torch.tensor(targets).to(device)
    )