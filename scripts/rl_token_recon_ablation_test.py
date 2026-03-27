import math

from scripts import rl_token_recon_ablation


def test_aggregate_batch_metrics_means_numeric_fields():
    metrics = rl_token_recon_ablation.aggregate_batch_metrics(
        [
            {
                "real_recon_loss": 1.0,
                "zero_recon_loss": 3.0,
                "shuffled_recon_loss": 2.0,
                "zero_recon_gap": 2.0,
                "shuffled_recon_gap": 1.0,
            },
            {
                "real_recon_loss": 5.0,
                "zero_recon_loss": 9.0,
                "shuffled_recon_loss": 7.0,
                "zero_recon_gap": 4.0,
                "shuffled_recon_gap": 2.0,
            },
        ]
    )

    assert math.isclose(metrics["real_recon_loss"], 3.0)
    assert math.isclose(metrics["zero_recon_loss"], 6.0)
    assert math.isclose(metrics["shuffled_recon_loss"], 4.5)
    assert math.isclose(metrics["zero_recon_gap"], 3.0)
    assert math.isclose(metrics["shuffled_recon_gap"], 1.5)


def test_aggregate_batch_metrics_excludes_per_batch_metadata():
    metrics = rl_token_recon_ablation.aggregate_batch_metrics(
        [
            {
                "batch_idx": 1,
                "batch_size": 8,
                "indices": [1, 2, 3],
                "real_recon_loss": 1.0,
                "zero_recon_gap": 2.0,
                "rl_token_pairwise_cosine_mean": 0.9,
            },
            {
                "batch_idx": 2,
                "batch_size": 8,
                "indices": [4, 5, 6],
                "real_recon_loss": 3.0,
                "zero_recon_gap": 4.0,
                "rl_token_pairwise_cosine_mean": 0.7,
            },
        ]
    )

    assert set(metrics) == {
        "real_recon_loss",
        "zero_recon_gap",
        "rl_token_pairwise_cosine_mean",
    }
    assert math.isclose(metrics["real_recon_loss"], 2.0)
    assert math.isclose(metrics["zero_recon_gap"], 3.0)
    assert math.isclose(metrics["rl_token_pairwise_cosine_mean"], 0.8)
