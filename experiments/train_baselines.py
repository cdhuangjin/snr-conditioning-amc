"""Train classic AMC baselines (CLDNN, LSTM) on the SAME split as the improved AWN,
and evaluate per-SNR + segments + macro F1 + kappa for fair comparison."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys, time, argparse, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.data_loader import Load_Dataset
from util.config import Config
from util.utils import fix_seed
from util.logger import create_logger
from train_snrc import dataset_split_indices, snr_to_bin, SnrDataset


def train_epoch(model, loader, criterion, optimizer, device, logger, epoch):
    model.train()
    meter_loss, meter_acc, n = 0.0, 0.0, 0
    for sig_batch, lab_batch, _ in tqdm(loader, desc=f'Epoch{epoch}', mininterval=0.3):
        sig_batch, lab_batch = sig_batch.to(device), lab_batch.to(device)
        logit, _ = model(sig_batch)
        loss = criterion(logit, lab_batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        acc = (torch.argmax(logit, 1) == lab_batch).double().mean().item()
        meter_loss += loss.item() * lab_batch.size(0); meter_acc += acc * lab_batch.size(0); n += lab_batch.size(0)
    return meter_loss / n, meter_acc / n


def val_epoch(model, loader, criterion, device):
    model.eval()
    meter_loss, meter_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for sig_batch, lab_batch, _ in loader:
            sig_batch, lab_batch = sig_batch.to(device), lab_batch.to(device)
            logit, _ = model(sig_batch)
            loss = criterion(logit, lab_batch)
            acc = (torch.argmax(logit, 1) == lab_batch).double().mean().item()
            meter_loss += loss.item() * lab_batch.size(0); meter_acc += acc * lab_batch.size(0); n += lab_batch.size(0)
    return meter_loss / n, meter_acc / n


class CLDNN(nn.Module):
    """CLDNN: CNN+LSTM+DFC, following West & O'Shea (2017) style, widely used as AMC baseline.
    BN+LeakyReLU are used to keep gradient flow stable (standard practice for this architecture)."""
    def __init__(self, num_classes=11, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(256, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)             # (B,256,32)
        x = x.transpose(1, 2)        # (B,32,256)
        x, _ = self.lstm(x)          # (B,32,128)
        x = x[:, -1, :]
        return self.fc(x), []


class LSTMNet(nn.Module):
    """Bidirectional single-layer LSTM with per-sample normalization (AMC standard baseline)."""
    def __init__(self, num_classes=11, hidden=128):
        super().__init__()
        self.norm = nn.InstanceNorm1d(2)
        self.lstm = nn.LSTM(2, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)        # (B,128,2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(self.drop(x)), []


def build_models():
    return {'cldnn': CLDNN, 'lstm': LSTMNet}


def eval_model(model, sig_test, lab_test, snr_test, device):
    model.eval()
    snr_levels = sorted(set(snr_test.tolist()))
    per_snr, preds_all = {}, []
    with torch.no_grad():
        for i in range(0, len(sig_test), 64):
            logit, _ = model(sig_test[i:i+64].to(device))
            preds_all.append(torch.argmax(logit, 1).cpu().numpy())
    preds_all = np.concatenate(preds_all)
    labels = lab_test.numpy()
    overall = float((labels == preds_all).mean())

    from sklearn.metrics import f1_score, cohen_kappa_score
    f1 = float(f1_score(labels, preds_all, average='macro'))
    kappa = float(cohen_kappa_score(labels, preds_all))

    snr_arr = np.array(snr_test)
    for bin_idx in snr_levels:
        sel = np.where(snr_arr == bin_idx)[0]
        per_snr[str(bin_idx * 2 - 20)] = float((labels[sel] == preds_all[sel]).mean())

    def seg(snrs):
        sel = np.where(np.isin(snr_arr, [snr_to_bin(s) for s in snrs]))[0]
        return float((labels[sel] == preds_all[sel]).mean()) if len(sel) else 0.0

    segs = {'low(<=-8dB)': seg(range(-20, -7, 2)), 'mid(-6~-2dB)': seg([-6, -4, -2]), 'high(>=0dB)': seg(range(0, 19, 2))}
    return overall, f1, kappa, segs, per_snr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cldnn', choices=['cldnn', 'lstm'])
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    fix_seed(args.seed)
    cfg = Config('2016.10a', train=True)
    device = torch.device(args.device)
    cfg.device = device

    tag = args.model
    exp_dir = os.path.join('experiments', f'2016.10a_{tag}')
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'log'), exist_ok=True)
    logger = create_logger(os.path.join(exp_dir, 'log', 'log.txt'))
    logger.info('=== Classic baseline training: {} ==='.format(tag))

    Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger)
    n_examples = Signals.shape[0]
    train_idx, val_idx, test_idx = dataset_split_indices(n_examples, mods, snrs)

    Set = pickle.load(open(os.path.join('data', 'RML2016.10a_dict.pkl'), 'rb'), encoding='bytes')
    snr_ref = np.array([snr for mod in mods for snr in snrs for _ in range(Set[(mod, snr)].shape[0])], dtype=np.float32)

    Signals_train, Labels_train = Signals[train_idx], Labels[train_idx]
    Signals_val, Labels_val = Signals[val_idx], Labels[val_idx]
    Signals_test, Labels_test = Signals[test_idx], Labels[test_idx]
    snr_test = snr_to_bin(snr_ref[test_idx]).astype(np.int64)

    train_ds = SnrDataset(Signals_train, Labels_train, torch.zeros(len(train_idx), dtype=torch.long))
    val_ds = SnrDataset(Signals_val, Labels_val, torch.zeros(len(val_idx), dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    model = build_models()[tag](num_classes=cfg.num_classes).to(device)
    logger.info('>>> total params: {:.3f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc, patience, early_counter = 0.0, cfg.patience, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch)
        vl, va = val_epoch(model, val_loader, criterion, device)
        logger.info('====> Epoch: {} Train Loss: {:.4f} Train acc: {:.4f} | Val Loss: {:.4f} Val acc: {:.4f} lr: {:.5f}'.format(
            epoch, tl, ta, vl, va, optimizer.param_groups[0]['lr']))
        if va >= best_val_acc:
            best_val_acc = va
            early_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, 'models', f'2016.10a_{tag.upper()}.pkl'))
        else:
            early_counter += 1
            if early_counter >= patience:
                logger.info('Early stopping'); break
        if early_counter and early_counter % cfg.milestone_step == 0:
            for g in optimizer.param_groups:
                g['lr'] *= cfg.gamma
            logger.info('  lr decay -> {:.5f}'.format(optimizer.param_groups[0]['lr']))
    logger.info('training time: {:.1f} min'.format((time.time() - t0) / 60))

    model.load_state_dict(torch.load(os.path.join(exp_dir, 'models', f'2016.10a_{tag.upper()}.pkl'), map_location='cpu'))
    overall, f1, kappa, segs, per_snr = eval_model(model, Signals_test, Labels_test, snr_test, device)
    logger.info('overall={:.4f} macroF1={:.4f} kappa={:.4f} seg={}'.format(overall, f1, kappa, segs))

    out = {'tag': tag, 'overall': overall, 'macro_f1': f1, 'kappa': kappa, 'seg': segs, 'per_snr': per_snr}
    with open(os.path.join('experiments', f'snr_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('DONE {} overall={:.4f} f1={:.4f} kappa={:.4f} seg={}'.format(tag, overall, f1, kappa, segs))


