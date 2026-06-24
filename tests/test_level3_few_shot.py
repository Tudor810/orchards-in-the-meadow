import torch

from agricultureai.data.episodic_sampler import (
    EpisodicBatchSampler,
    episodic_collate_fn,
    map_raw_to_level3,
)
from agricultureai.models.proto_net import (
    PrototypicalNetwork,
    compute_prototypes,
    prototypical_loss,
    squared_euclidean_distance,
)


def test_raw_to_level3_mapping_orders_anchors_then_anomalies():
    # Rejection anchors first (meadows), then true anomalies.
    assert map_raw_to_level3(7) == 0  # permanent meadows
    assert map_raw_to_level3(8) == 1  # temporary meadows
    assert map_raw_to_level3(5) == 2  # orchards
    assert map_raw_to_level3(6) == 3  # nut trees
    assert map_raw_to_level3(4) == 4  # sunflowers


def test_episodic_sampler_emits_support_then_query_blocks():
    # 5 classes, 40 samples each, comfortably above k+q.
    labels = [c for c in range(5) for _ in range(40)]
    sampler = EpisodicBatchSampler(
        labels, n_way=5, k_shot=5, q_query=15, num_episodes=10, seed=0
    )

    assert sampler.q_query == 15  # no scale-down needed
    episodes = list(iter(sampler))
    assert len(episodes) == 10

    for episode in episodes:
        assert len(episode) == 5 * (5 + 15)
        # All indices unique within an episode (disjoint support/query).
        assert len(set(episode)) == len(episode)


def test_episodic_sampler_gracefully_scales_down_q_query():
    # Class 2 (orchards) is the bottleneck with only 8 samples.
    labels = [0] * 50 + [1] * 50 + [2] * 8 + [3] * 50 + [4] * 50
    sampler = EpisodicBatchSampler(
        labels, n_way=5, k_shot=5, q_query=15, num_episodes=5, seed=1
    )

    # q_eff = min(15, 8 - 5) = 3.
    assert sampler.q_query == 3
    for episode in list(iter(sampler)):
        assert len(episode) == 5 * (5 + 3)


def test_episodic_collate_assigns_relative_labels():
    # Build a fake fetched episode: 2-way, 2-shot, 3-query.
    n_way, k_shot, q_query = 2, 2, 3
    seq_len, channels = 4, 3
    n_support = n_way * k_shot
    n_query = n_way * q_query

    batch = [
        (torch.full((seq_len, channels), float(i)), -1)
        for i in range(n_support + n_query)
    ]
    support_x, support_y, query_x, query_y = episodic_collate_fn(
        batch, n_way=n_way, k_shot=k_shot, q_query=q_query
    )

    assert support_x.shape == (n_support, seq_len, channels)
    assert query_x.shape == (n_query, seq_len, channels)
    assert support_y.tolist() == [0, 0, 1, 1]
    assert query_y.tolist() == [0, 0, 0, 1, 1, 1]


def test_squared_euclidean_distance_matches_manual():
    queries = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    prototypes = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

    dist = squared_euclidean_distance(queries, prototypes)

    assert dist.shape == (2, 2)
    assert torch.allclose(dist[0, 0], torch.tensor(0.0))
    assert torch.allclose(dist[0, 1], torch.tensor(25.0))  # 3^2 + 4^2
    assert torch.allclose(dist[1, 1], torch.tensor(0.0))


def test_compute_prototypes_averages_per_class():
    embeddings = torch.tensor(
        [[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [12.0, 12.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])

    protos = compute_prototypes(embeddings, labels, n_way=2)

    assert torch.allclose(protos[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(protos[1], torch.tensor([11.0, 11.0]))


def test_prototypical_loss_perfect_separation_is_accurate():
    # Queries sit exactly on their class prototype -> 100% accuracy.
    query_emb = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    query_y = torch.tensor([0, 1])
    prototypes = torch.tensor([[0.0, 0.0], [10.0, 10.0]])

    out = prototypical_loss(query_emb, query_y, prototypes)

    assert out.log_p_y.shape == (2, 2)
    assert torch.isclose(out.accuracy, torch.tensor(1.0))
    assert out.loss.item() >= 0.0


def test_prototypical_network_episode_shapes_and_backward():
    torch.manual_seed(0)
    n_way, k_shot, q_query = 5, 5, 15
    seq_len, channels = 45, 13
    model = PrototypicalNetwork(in_channels=channels, hidden_dim=64, encoder="cnn")

    support_x = torch.randn(n_way * k_shot, seq_len, channels)
    support_y = torch.arange(n_way).repeat_interleave(k_shot)
    query_x = torch.randn(n_way * q_query, seq_len, channels)
    query_y = torch.arange(n_way).repeat_interleave(q_query)

    out = model.proto_episode(support_x, support_y, query_x, query_y, n_way)

    assert out.log_p_y.shape == (n_way * q_query, n_way)
    assert out.prototypes.shape == (n_way, 64)
    # Loss is differentiable w.r.t. encoder params.
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())
