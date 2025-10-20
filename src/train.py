import math
import torch
import csv, os
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from .config import (SEED, DATA_DIR, PROCESSED_PT, BATCH_SIZE, EPOCHS, LR, WD, LAMBDA_LAP)
from .dataset import VISDataset
from .model_pignn import PIGNN, loss_with_physics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = VISDataset(DATA_DIR, regenerate=True)

    n = len(ds)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    in_dim  = ds[0].x.size(1)
    # meta is [1, meta_dim] per sample → batch becomes [B, meta_dim]
    meta_dim = ds[0].meta.size(-1) if hasattr(ds[0], 'meta') else 0
    model = PIGNN(in_dim=in_dim, meta_dim=meta_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best = float('inf')

    FIG_DIR = DATA_DIR / "fig"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_CSV = DATA_DIR / "metrics.csv"

    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_mse_nadir","val_mse_rocof","val_mse_tsettle"])

    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred, node_emb = model(batch)
            loss = loss_with_physics(pred, batch.y, node_emb, batch.edge_index, batch.edge_attr, LAMBDA_LAP)
            loss.backward(); opt.step()
            running += float(loss.detach().cpu())
        train_loss = running / max(1, len(train_loader))

        # validation
        model.eval()
        with torch.no_grad():
            mse = torch.zeros(3, device=device)
            n_graphs = 0
            n_tsettle = 0
            for batch in val_loader:
                batch = batch.to(device)
                pred, _ = model(batch)
                mse[0] += torch.sum((pred[:,0] - batch.y[:,0])**2)  # nadir
                mse[1] += torch.sum((pred[:,1] - batch.y[:,1])**2)  # RoCoF
                m = (batch.y[:,2] >= 0)
                if m.any():
                    mse[2] += torch.sum((pred[m,2] - batch.y[m,2])**2)
                    n_tsettle += int(m.sum())
                n_graphs += batch.num_graphs

            val_metrics = dict(
                mse_nadir = float(mse[0].cpu())/max(1, n_graphs),
                mse_rocof = float(mse[1].cpu())/max(1, n_graphs),
                mse_tsettle = float(mse[2].cpu())/max(1, n_tsettle) if n_tsettle>0 else float('nan'),
            )
        
        with open(METRICS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{train_loss:.6e}",
                        f"{val_metrics['mse_nadir']:.6e}",
                        f"{val_metrics['mse_rocof']:.6e}",
                        f"{val_metrics['mse_tsettle'] if not math.isnan(val_metrics['mse_tsettle']) else float('nan'):.6e}"])

        score = val_metrics['mse_nadir'] + val_metrics['mse_rocof'] + (val_metrics['mse_tsettle'] if not math.isnan(val_metrics['mse_tsettle']) else 0.0)
        print(f"[{ep:03d}] train={train_loss:.4e} | val={val_metrics}")

        if score < best:
            best = score
            torch.save(model.state_dict(), DATA_DIR / "pignn_best.pt")

    # Test
    model.load_state_dict(torch.load(DATA_DIR / "pignn_best.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        mse = torch.zeros(3, device=device)
        n_graphs = 0
        n_tsettle = 0
        for batch in DataLoader(test_ds, batch_size=BATCH_SIZE):
            batch = batch.to(device)
            pred, _ = model(batch)
            mse[0] += torch.sum((pred[:,0] - batch.y[:,0])**2)
            mse[1] += torch.sum((pred[:,1] - batch.y[:,1])**2)
            m = (batch.y[:,2] >= 0)
            if m.any():
                mse[2] += torch.sum((pred[m,2] - batch.y[m,2])**2)
                n_tsettle += int(m.sum())
            n_graphs += batch.num_graphs

        test_metrics = dict(
            mse_nadir = float(mse[0].cpu())/max(1, n_graphs),
            mse_rocof = float(mse[1].cpu())/max(1, n_graphs),
            mse_tsettle = float(mse[2].cpu())/max(1, n_tsettle) if n_tsettle>0 else float('nan'),
        )
    print("\n=== TEST ===")
    print(test_metrics)

if __name__ == "__main__":
    main()
