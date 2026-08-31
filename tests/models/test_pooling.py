import torch

from vaak.models.backends.pooling import AttentiveStatisticsPooling, MeanPooling


def test_mean_pooling() -> None:
    batch_size = 2
    time_steps = 50
    feature_dim = 768

    features = torch.randn(batch_size, time_steps, feature_dim)
    pooler = MeanPooling()

    pooled = pooler(features)

    assert pooled.shape == (batch_size, feature_dim)


def test_attentive_statistics_pooling() -> None:
    batch_size = 2
    time_steps = 50
    feature_dim = 768

    features = torch.randn(batch_size, time_steps, feature_dim)
    pooler = AttentiveStatisticsPooling(input_dim=feature_dim, attention_dim=128)

    pooled = pooler(features)

    # ASP concatenates mean and std, so output dim must be 2 * feature_dim
    assert pooled.shape == (batch_size, feature_dim * 2)

    # Ensure no NaNs are produced (e.g., from negative variance edge cases)
    assert not torch.isnan(pooled).any()
