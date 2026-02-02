"""モジュリングモジュールのテスト（GPUなし、APIモック対応）"""

import numpy as np
from signate_studentcup_2025.modeling.evaluate import compute_metrics


def test_compute_metrics_accuracy():
    """Accuracy計算のテスト"""
    predictions = [(1, 2), (3, 4), (5, 6)]
    ground_truth = [(1, 2), (4, 3), (6, 5)]  # 順序は違うがセットは同じ

    metrics = compute_metrics(predictions, ground_truth)

    # 全て正解（順序は問わない）
    assert metrics["accuracy"] == 1.0


def test_compute_metrics_partial_match():
    """部分一致の場合のテスト"""
    predictions = [(1, 2), (3, 5), (7, 8)]
    ground_truth = [(1, 2), (3, 4), (6, 5)]

    metrics = compute_metrics(predictions, ground_truth)

    # 1つ目のみ完全一致
    assert metrics["accuracy"] == 1/3


def test_compute_metrics_hit_rate():
    """Hit Rate計算のテスト"""
    predictions = [(1, 10), (2, 10), (3, 10)]
    ground_truth = [(1, 2), (2, 3), (3, 4)]

    metrics = compute_metrics(predictions, ground_truth, k_values=[1, 5, 10])

    # Hit Rate@1: 最初の1件に正解が含まれる確率
    assert "hit_rate@1" in metrics

    # Hit Rate@10: 上位10件に正解が含まれる確率
    assert "hit_rate@10" in metrics


def test_compute_metrics_mrr():
    """MRR計算のテスト"""
    predictions = [(1, 2), (2, 3), (10, 20)]
    ground_truth = [(1, 2), (2, 4), (5, 6)]

    metrics = compute_metrics(predictions, ground_truth)

    # 1つ目は1位で正解、2つ目は1位で部分正解、3つ目は不正解
    # MRR = (1/1 + 1/1 + 0) / 3 = 2/3
    assert metrics["mrr"] > 0
