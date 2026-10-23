"""Fine-tune MobileNetV3-Large on SCUT-FBP5500 face crops (multi-task: beauty score + sex).

Crops use the same landmark-bbox rule as the browser (mogscan_metrics.crop_box), so the
model sees the same framing live as in training. 15% of the official training split is held
out as a validation set for early stopping and for fitting the fusion weight later.

Outputs:
  out/cnn.pt            best weights (by validation Pearson r)
  out/cnn_preds.csv     val + test predictions (score, P(male))
  out/results_cnn.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from mogscan_metrics import crop_box

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "SCUT-FBP5500_v2" / "Images"
OUT = ROOT / "out"
SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_crops():
    d = np.load(OUT / "landmarks.npz")
    crops = {}
    for name, lm, (w, h) in zip(d["names"], d["landmarks"], d["sizes"]):
        x, y, side = crop_box(lm[:, :2] * np.array([w, h]), w, h)
        crops[str(name)] = (x, y, x + side, y + side)
    return crops


class FaceSet(Dataset):
    def __init__(self, df, crops, train):
        self.df, self.crops = df.reset_index(drop=True), crops
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomAffine(8, scale=(0.93, 1.07), translate=(0.04, 0.04)),
               transforms.ColorJitter(0.3, 0.3, 0.2, 0.03)] if train else []
        self.tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), *aug,
                                      transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(DATA / r["name"]).convert("RGB").crop(tuple(int(round(v)) for v in self.crops[r["name"]]))
        return self.tf(img), torch.tensor(r.score, dtype=torch.float32), torch.tensor(float(r.sex == "M"))


class MogNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        base.classifier = nn.Sequential(nn.Linear(960, 512), nn.Hardswish(), nn.Dropout(0.3), nn.Linear(512, 2))
        self.net = base

    def forward(self, x):
        out = self.net(x)
        return out[:, 0] + 3.0, out[:, 1]  # score centred on the dataset mean, sex logit


@torch.no_grad()
def predict(model, loader):
    model.eval()
    s, m = [], []
    for x, _, _ in loader:
        score, sex = model(x.to(DEVICE))
        s.append(score.cpu()); m.append(torch.sigmoid(sex).cpu())
    return torch.cat(s).numpy(), torch.cat(m).numpy()


def main(epochs=25):
    torch.manual_seed(0)
    df = pd.read_csv(OUT / "features.csv")
    crops = load_crops()
    train_all, test = df[df.split == "train"], df[df.split == "test"]
    val = train_all.sample(frac=0.15, random_state=0)
    train = train_all.drop(val.index)
    dl = lambda d, t: DataLoader(FaceSet(d, crops, t), batch_size=32, shuffle=t, num_workers=4,
                                 persistent_workers=True)
    tr_dl, va_dl, te_dl = dl(train, True), dl(val, False), dl(test, False)

    model = MogNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(tr_dl))
    best = -1.0
    for ep in range(epochs):
        model.train()
        for x, y, sex in tr_dl:
            x, y, sex = x.to(DEVICE), y.to(DEVICE), sex.to(DEVICE)
            score, logit = model(x)
            loss = nn.functional.smooth_l1_loss(score, y) + 0.2 * nn.functional.binary_cross_entropy_with_logits(logit, sex)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ps, pm = predict(model, va_dl)
        r = pearsonr(val.score, ps)[0]
        acc = np.mean((pm > 0.5) == (val.sex == "M").values)
        print(f"epoch {ep + 1}: val r={r:.4f} sex acc={acc:.3f}", flush=True)
        if r > best:
            best = r
            torch.save(model.state_dict(), OUT / "cnn.pt")

    model.load_state_dict(torch.load(OUT / "cnn.pt"))
    vs, vm = predict(model, va_dl)
    ts, tm = predict(model, te_dl)
    pd.concat([pd.DataFrame({"name": val.name.values, "set": "val", "cnn": vs, "p_male": vm}),
               pd.DataFrame({"name": test.name.values, "set": "test", "cnn": ts, "p_male": tm})]
              ).to_csv(OUT / "cnn_preds.csv", index=False)
    res = {"pearson": float(pearsonr(test.score, ts)[0]), "mae": float(np.mean(np.abs(test.score - ts))),
           "rmse": float(np.sqrt(np.mean((test.score - ts) ** 2))),
           "sex_acc": float(np.mean((tm > 0.5) == (test.sex == "M").values)), "best_val_pearson": float(best)}
    json.dump(res, open(OUT / "results_cnn.json", "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()
