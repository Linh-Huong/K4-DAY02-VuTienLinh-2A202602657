"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.
    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    """
    images_dir = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total = n_train + n_val + n_test

    # 1. Kiểm tra tổng số ảnh
    assert total == 17509, f"Tổng số ảnh phải đúng bằng 17.509, thực tế là {total}"

    # 2. Kiểm tra giao rỗng
    s_train = set(train_df["Filename"])
    s_val = set(val_df["Filename"])
    s_test = set(test_df["Filename"])

    inter_train_val = s_train & s_val
    inter_train_test = s_train & s_test
    inter_val_test = s_val & s_test

    assert len(inter_train_val) == 0, f"Giao giữa train và val không rỗng: {len(inter_train_val)} ảnh"
    assert len(inter_train_test) == 0, f"Giao giữa train và test không rỗng: {len(inter_train_test)} ảnh"
    assert len(inter_val_test) == 0, f"Giao giữa val và test không rỗng: {len(inter_val_test)} ảnh"

    # 3. Hợp đủ 17509
    union_all = s_train | s_val | s_test
    assert len(union_all) == 17509, f"Hợp ba tập phải bằng 17.509 ảnh, thực tế {len(union_all)}"

    # 4. Kiểm tra mọi file tồn tại trong images_dir
    missing_files = []
    for fn in union_all:
        if not (images_dir / fn).is_file():
            missing_files.append(fn)
            if len(missing_files) >= 5:
                break
    assert len(missing_files) == 0, f"Có ảnh trong CSV không tồn tại trong thư mục ảnh: {missing_files[:5]}"

    # Thống kê phân bố lớp
    per_class = {
        "train": train_df["Label"].value_counts().sort_index().to_dict(),
        "val": val_df["Label"].value_counts().sort_index().to_dict(),
        "test": test_df["Label"].value_counts().sort_index().to_dict(),
    }

    stats = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total},
        "per_class": per_class,
        "overlap": {
            "train_val": len(inter_train_val),
            "train_test": len(inter_train_test),
            "val_test": len(inter_val_test),
        },
        "all_files_exist": True,
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> T.Compose:
    """Tạo torchvision transforms theo `train` và `aug`."""
    if train:
        transform_list = []
        if aug == "basic":
            transform_list = [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
            ]
        elif aug == "color":
            transform_list = [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        elif aug == "randaug":
            transform_list = [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandAugment(num_ops=2, magnitude=9),
            ]
        elif aug == "trivial":
            transform_list = [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.TrivialAugmentWide(),
            ]
        else:
            transform_list = [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
            ]

        transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return T.Compose(transform_list)
    else:
        if img_size == 256:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fn = self.filenames[i]
        path = self.images_dir / fn
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[i]
        return img, label, fn


def seed_worker(worker_id: int) -> None:
    """Worker init function để cố định seed khi dùng nhiều worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: Optional[str] = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    ds = DeepWeedsDataset(df=df, images_dir=images_dir, transform=transform)

    data_sampler = None
    shuffle = False

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / class_counts[label] for label in df["Label"]]
            data_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            shuffle = True
    else:
        shuffle = False

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=data_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    return loader


