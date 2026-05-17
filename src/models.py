import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedGraphNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_in = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim*2, hidden_dim)
        self.W_z = nn.Linear(hidden_dim*2, hidden_dim)
        self.W_h = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, adj, nodes):
        in_msg = torch.bmm(adj, nodes)
        out_msg = torch.bmm(adj.transpose(1, 2), nodes)
        a = self.W_in(in_msg) + self.W_out(out_msg)
        combined = torch.cat([a, nodes], 2)
        r = torch.sigmoid(self.W_r(combined))
        z = torch.sigmoid(self.W_z(combined))
        h_hat = torch.tanh(self.W_h(combined))
        return (1 - z) * nodes + z * h_hat

class SRGNN_Ultimate(nn.Module):
    def __init__(self, n_items, n_cats, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embeddings
        self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, hidden_dim, padding_idx=0)   
        self.event_emb = nn.Embedding(4, hidden_dim)                     

        # GNN Engine
        self.gnn = GatedGraphNN(hidden_dim)

        # Attention Mechanism
        self.att_w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_v = nn.Linear(hidden_dim, 1, bias=False)

        # Final Layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_items, bias=False)

    def forward(self, adj, nodes, cats, evts, last_idx, masks):
        # 1. GNN Katmanı üzerinden Nesne Temsili Öğrenimi
        h = self.item_emb(nodes) 
        h = self.gnn(adj, h)

        # 2. Lokal Oturum Temsili (Son Tıklanan Ürün)
        last_idx_ex = last_idx.view(-1, 1, 1).expand(-1, -1, self.hidden_dim)
        s_local = h.gather(1, last_idx_ex).squeeze(1) 

        # 🔥 FEATURE FUSION: Kategori ve Aksiyon Ağırlıklarının Enjeksiyonu
        h_cat = self.cat_emb(cats)
        h_evt = self.event_emb(evts)
        s_local_enhanced = s_local + (h_cat * 0.5) + (h_evt * 0.5) 

        # 3. Global Oturum Temsili (Self-Attention ile Gürültü Filtreleme)
        q = self.att_w1(h)
        k = self.att_w2(s_local_enhanced).unsqueeze(1)
        alpha = torch.sigmoid(self.att_v(torch.tanh(q + k)))
        alpha = alpha.masked_fill(~masks.unsqueeze(-1), 0)
        s_global = torch.sum(alpha * h, 1)

        # 4. Tahmin Katmanı
        s_final = self.fc1(s_global) + self.fc2(s_local_enhanced)
        s_final = self.dropout(s_final)
        scores = self.output(s_final)
        return scores