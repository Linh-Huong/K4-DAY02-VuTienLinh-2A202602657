"""Kiểm tra toàn diện các module trong thư mục code/."""
import math
import sys
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

import dataset
import model as model_utils
import losses
import inference
import benchmark


class TestCodePipeline(unittest.TestCase):
    def test_loss_focal_gamma_zero_equals_ce(self):
        """Focal loss với gamma=0 phải bằng đúng Cross-Entropy (sai số < 1e-6)."""
        torch.manual_seed(42)
        logits = torch.randn(10, 9)
        targets = torch.randint(0, 9, (10,))

        ce_loss = nn.CrossEntropyLoss()(logits, targets)
        focal_loss = losses.FocalLoss(gamma=0.0)(logits, targets)

        self.assertAlmostEqual(ce_loss.item(), focal_loss.item(), places=5)

    def test_loss_label_smoothing_zero_equals_ce(self):
        """Label smoothing CE với eps=0 phải bằng đúng Cross-Entropy."""
        torch.manual_seed(42)
        logits = torch.randn(10, 9)
        targets = torch.randint(0, 9, (10,))

        ce_loss = nn.CrossEntropyLoss()(logits, targets)
        ls_loss = losses.LabelSmoothingCE(smoothing=0.0)(logits, targets)

        self.assertAlmostEqual(ce_loss.item(), ls_loss.item(), places=5)

    def test_class_weights(self):
        counts = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
        w = losses.class_weights(counts, beta=0.0)
        self.assertEqual(len(w), 9)
        # Lớp Negatives nhiều ảnh nhất nên trọng số phải nhỏ nhất
        self.assertEqual(int(torch.argmin(w)), 8)

    def test_mixup_and_cutmix(self):
        torch.manual_seed(42)
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 3])

        # Mixup
        x_m, (y_a, y_b, lam) = losses.mix_batch(x, y, alpha=1.0, mode="mixup")
        self.assertEqual(x_m.shape, x.shape)
        self.assertTrue(0.0 <= lam <= 1.0)

        # CutMix
        x_c, (y_ca, y_cb, lam_c) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix")
        self.assertEqual(x_c.shape, x.shape)
        self.assertTrue(0.0 <= lam_c <= 1.0)

    def test_model_build_and_freeze(self):
        m = model_utils.build_model("resnet50", pretrained=False, num_classes=9, init="finetune")
        p_count = model_utils.count_params(m)
        self.assertGreater(p_count, 20.0)  # ResNet50 ~25M params

        groups = model_utils.param_groups(m, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
        self.assertGreaterEqual(len(groups), 2)

        # Test frozen mode
        m_frozen = model_utils.build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
        trainable = [p for p in m_frozen.parameters() if p.requires_grad]
        # Chỉ head mới train được
        self.assertEqual(len(trainable), 2)  # weight + bias của linear head

    def test_dataset_split_and_check(self):
        labels_dir = ROOT / "data" / "labels"
        images_dir = ROOT / "data" / "images"
        if not (labels_dir.exists() and images_dir.exists()):
            self.skipTest("Thư mục data chưa có sẵn để kiểm tra")

        train_df, val_df, test_df = dataset.load_split(labels_dir, fold=0)
        stats = dataset.check_split(train_df, val_df, test_df, images_dir)
        self.assertEqual(stats["n"]["total"], 17509)
        self.assertEqual(stats["overlap"]["train_val"], 0)
        self.assertEqual(stats["overlap"]["train_test"], 0)
        self.assertEqual(stats["overlap"]["val_test"], 0)

    def test_inference_and_temperature_scaling(self):
        rng = np.random.default_rng(42)
        logits = rng.normal(size=(100, 9))
        y = rng.integers(0, 9, 100)

        t = inference.fit_temperature(logits, y)
        self.assertGreater(t, 0.0)

        probs = inference.apply_temperature(logits, t)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(100), atol=1e-5)

        # Test view aggregation
        agg = inference.aggregate_views([logits, logits], space="prob")
        self.assertEqual(agg.shape, (100, 9))
        np.testing.assert_allclose(agg.sum(axis=1), np.ones(100), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
