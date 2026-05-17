import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from functools import partial

from data_pipeline import load_and_preprocess_data, create_mappings_and_split, process_sessions
from dataset import EnhancedSessionDataset, collate_fn
from models import SRGNN_Ultimate

def main():
    # --- Hiperparametre Konfigürasyonu ---
    DATA_DIR = "data"
    EVENTS_PATH = os.path.join(DATA_DIR, "events.csv")
    PROP1_PATH = os.path.join(DATA_DIR, "item_properties_part1.csv")
    PROP2_PATH = os.path.join(DATA_DIR, "item_properties_part2.csv")

    BATCH_SIZE = 128
    EPOCHS = 20
    HIDDEN_DIM = 256
    DROPOUT = 0.4
    LR = 0.001
    WEIGHT_DECAY = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running execution on device: {device}")

    # --- Veri Boru Hattı ---
    events = load_and_preprocess_data(EVENTS_PATH, PROP1_PATH, PROP2_PATH)
    train_df, val_df, test_df, n_items, n_cats = create_mappings_and_split(events)

    print("\nGenerating Datasets...")
    train_data = process_sessions(train_df, augment=True) # Veri Çoğaltma Aktif
    val_data = process_sessions(val_df, augment=False)
    test_data = process_sessions(test_df, augment=False)

    # --- DataLoader Yapılandırması ---
    custom_collate = partial(collate_fn, device=device)
    train_loader = DataLoader(EnhancedSessionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(EnhancedSessionDataset(val_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(EnhancedSessionDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    # --- Model Kurulumu ---
    model = SRGNN_Ultimate(n_items, n_cats, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_recall = 0
    patience_limit = 5
    patience_counter = 0

    print("\n🚀 Starting Training (Ultimate Version)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            adj, nodes, cats, evts, lasts, masks, targets = batch

            optimizer.zero_grad()
            scores = model(adj, nodes, cats, evts, lasts, masks)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Loss: {total_loss/len(train_loader):.4f}")

        # --- Doğrulama (Validation) Döngüsü ---
        model.eval()
        hits, ranks = [], []
        with torch.no_grad():
            for batch in val_loader:
                adj, nodes, cats, evts, lasts, masks, targets = batch
                scores = model(adj, nodes, cats, evts, lasts, masks)
                sub_scores = scores.topk(20)[1].cpu().numpy()
                targets = targets.cpu().numpy()

                for score, target in zip(sub_scores, targets):
                    if target in score:
                        hits.append(1)
                        ranks.append(1 / (np.where(score == target)[0][0] + 1))
                    else:
                        hits.append(0)
                        ranks.append(0)

        val_recall = np.mean(hits) * 100
        val_mrr = np.mean(ranks)
        print(f"  🔎 Val Recall@20: {val_recall:.2f}% | MRR@20: {val_mrr:.4f}")

        scheduler.step(val_recall)

        # Early Stopping (Erken Durdurma) Kontrolü
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), "SRGNN_Ultimate.pth")
            patience_counter = 0
            print("  ✅ Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("  🛑 Early Stopping Triggered")
                break

    # --- Son Değerlendirme (Test Seti) ---
    print("\n🧪 Executing Final Testing Phase...")
    if os.path.exists("SRGNN_Ultimate.pth"):
        model.load_state_dict(torch.load("SRGNN_Ultimate.pth"))
    model.eval()
    hits = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            adj, nodes, cats, evts, lasts, masks, targets = batch
            scores = model(adj, nodes, cats, evts, lasts, masks)
            sub_scores = scores.topk(20)[1].cpu().numpy()
            targets = targets.cpu().numpy()
            for score, target in zip(sub_scores, targets):
                hits.append(1 if target in score else 0)

    print(f"🏆 Final Test Recall@20: {np.mean(hits)*100:.2f}%")

if __name__ == "__main__":
    main()