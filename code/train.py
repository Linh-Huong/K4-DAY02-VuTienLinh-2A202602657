"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình: đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, fields
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# Đảm bảo import được code và eval.py
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from eval import save_predictions, compute_metrics, NUM_CLASSES
import dataset
import model as model_utils
import losses


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = model_utils.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.clone().detach()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    # Nếu backbone đóng băng: giữ BatchNorm ở chế độ eval
    if getattr(model, "_frozen_backbone", False):
        for m in model.modules():
            if isinstance(m, (nn.modules.batchnorm._BatchNorm,)):
                m.eval()

    total_loss = 0.0
    total_samples = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        b = x.size(0)

        # Mixup / CutMix
        if cfg.mix in ["mixup", "cutmix"]:
            x_mixed, targets = losses.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
            with autocast(device_type="cuda", enabled=cfg.amp and device.type == "cuda"):
                out = model(x_mixed)
                loss = losses.mixed_loss(criterion, out, targets)
        else:
            with autocast(device_type="cuda", enabled=cfg.amp and device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)

        optimizer.zero_grad()
        if cfg.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()


        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * b
        total_samples += b

    avg_loss = total_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên loader ở chế độ eval, trả về (filenames, y_true, logits, avg_loss)."""
    model.eval()
    filenames = []
    y_true_list = []
    logits_list = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for x, y, fns in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            b = x.size(0)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available() and device.type == "cuda"):
                logits = model(x)
                if criterion is not None:
                    loss = criterion(logits, y)
                    total_loss += loss.item() * b

            filenames.extend(fns)
            y_true_list.append(y.cpu().numpy())
            logits_list.append(logits.cpu().float().numpy())
            total_samples += b

    all_y_true = np.concatenate(y_true_list, axis=0)
    all_logits = np.concatenate(logits_list, axis=0)
    avg_loss = total_loss / max(1, total_samples)
    return filenames, all_y_true, all_logits, avg_loss


def softmax_np(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    val_acc = [h["val_top1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue", marker="o", markersize=3)
    ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Metric plot
    ax2.plot(epochs, val_f1, label="Val Macro-F1", color="tab:green", marker="^", markersize=3)
    ax2.plot(epochs, val_acc, label="Val Top-1 Acc", color="tab:purple", marker="x", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{title} - Val Metrics")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Thư mục chạy và config.json
    out_dir = run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 2. Đọc split & check split
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. DataLoaders
    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tf,
        batch_size=cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tf,
        batch_size=cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 4. Model & Criterion
    model = model_utils.build_model(
        name=cfg.backbone,
        num_classes=NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    ).to(device)

    # Trọng số lớp nếu có
    weight = None
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().to_dict()
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        weight = losses.class_weights(counts, beta=beta).to(device)

    criterion = losses.build_criterion(
        kind=cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weight,
    )
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    ema = None
    if cfg.ema_decay is not None and cfg.ema_decay > 0:
        ema = EMA(model, decay=cfg.ema_decay)

    # 5. Training loop
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_ckpt_path = out_dir / "best_model.pth"
    start_train_time = time.time()
    epoch_times = []

    print(f"\n[{cfg.exp_id}] Bắt đầu huấn luyện backbone={cfg.backbone}, init={cfg.init}, loss={cfg.loss}, seed={cfg.seed}...")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_info = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            device=device,
            ema=ema
        )
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Đánh giá trên Val
        if ema is not None:
            ema.apply_shadow(model)

        val_fns, val_y_true, val_logits, val_loss = evaluate(model, val_loader, eval_criterion, device)
        val_probs = softmax_np(val_logits)
        val_y_pred = val_probs.argmax(axis=1)
        val_metrics = compute_metrics(val_y_true, val_y_pred, val_probs)
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_top1 = float(val_metrics["top1"])

        if ema is not None:
            ema.restore(model)

        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} [{t_epoch:.1f}s] - Train Loss: {train_info['train_loss']:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_macro_f1*100:.2f}% | Val Acc: {val_top1*100:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_info["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_top1": val_top1,
            "lr": train_info["lr"],
            "epoch_sec": t_epoch,
        })

        # Lưu checkpoint tốt nhất theo Macro-F1 val (hòa thì lấy epoch sớm hơn)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            # Lưu state_dict
            if ema is not None:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), best_ckpt_path)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), best_ckpt_path)

    total_train_sec = time.time() - start_train_time
    avg_epoch_sec = float(np.mean(epoch_times))

    # 6. Load checkpoint tốt nhất và xuất dự đoán trên VAL
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    val_fns, val_y_true, val_logits, _ = evaluate(model, val_loader, eval_criterion, device)
    val_probs = softmax_np(val_logits)
    save_predictions(pred_path(cfg, "val"), val_fns, val_y_true, val_probs)
    np.save(out_dir / "val_logits.npy", val_logits)

    # 7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): Đánh giá test ĐÚNG MỘT LẦN
    test_metrics_summary = None
    if cfg.save_test_predictions:
        test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_tf,
            batch_size=cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_fns, test_y_true, test_logits, _ = evaluate(model, test_loader, eval_criterion, device)
        test_probs = softmax_np(test_logits)
        test_y_pred = test_probs.argmax(axis=1)
        save_predictions(pred_path(cfg, "test"), test_fns, test_y_true, test_probs)
        np.save(out_dir / "test_logits.npy", test_logits)
        raw_test_metrics = compute_metrics(test_y_true, test_y_pred, test_probs)
        test_metrics_summary = {
            "top1": float(raw_test_metrics["top1"]),
            "macro_f1": float(raw_test_metrics["macro_f1"]),
            "balanced_acc": float(raw_test_metrics["balanced_acc"]),
            "ece": float(raw_test_metrics["ece"]),
            "nll": float(raw_test_metrics["nll"]),
        }
        print(f"[{cfg.exp_id}] KẾT QUẢ TEST: Top-1={test_metrics_summary['top1']*100:.2f}%, Macro-F1={test_metrics_summary['macro_f1']*100:.2f}%")

    # 8. Ghi history.csv và vẽ biểu đồ training
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    curve_path = Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"{cfg.exp_id} - {cfg.backbone}")

    n_params = model_utils.count_params(model)
    n_gmacs = model_utils.count_gmacs(model, img_size=cfg.img_size)

    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "val_top1": history[best_epoch - 1]["val_top1"] if best_epoch > 0 else 0.0,
        "avg_epoch_sec": avg_epoch_sec,
        "total_train_sec": total_train_sec,
        "params_m": n_params,
        "gmacs": n_gmacs,
        "test_metrics": test_metrics_summary,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Ép kiểu tham số theo các trường của Config."""
    valid_fields = {f.name: f.type for f in fields(Config)}
    overrides = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Tham số không hợp lệ: {p} (phải có dạng key=value)")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in valid_fields:
            raise KeyError(f"Trường cấu hình '{k}' không tồn tại trong Config. Các trường hợp lệ: {list(valid_fields.keys())}")

        if v.lower() == "none":
            overrides[k] = None
        elif v.lower() == "true":
            overrides[k] = True
        elif v.lower() == "false":
            overrides[k] = False
        else:
            # Ép kiểu int/float nếu được
            try:
                if "." in v or "e" in v.lower():
                    overrides[k] = float(v)
                else:
                    overrides[k] = int(v)
            except ValueError:
                overrides[k] = v
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds.")
    parser.add_argument("--set", nargs="*", default=[], help="Cấu hình ghi đè dạng key=value")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    res = run(cfg)
    print("\nHoàn thành thí nghiệm:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
