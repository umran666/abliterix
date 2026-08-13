"""Router expert-id extraction across MoE family tuple orders."""

import torch

from abliterix.core.engine import extract_router_expert_ids


def test_bailing_v3_indices_first():
    idx = torch.tensor([[0, 3, 7]], dtype=torch.int64)
    weight = torch.tensor([[0.5, 0.3, 0.2]])
    logits = torch.randn(1, 16)
    got = extract_router_expert_ids((idx, weight, logits))
    assert torch.equal(got, idx)


def test_legacy_indices_last_still_prefers_int_tensor():
    weight = torch.tensor([[0.5, 0.3, 0.2]])
    logits = torch.randn(1, 16)
    idx = torch.tensor([[1, 2, 4]], dtype=torch.int32)
    got = extract_router_expert_ids((weight, logits, idx))
    assert torch.equal(got, idx)


def test_two_tuple_keeps_second():
    weight = torch.randn(2, 8)
    idx = torch.tensor([[2, 5]], dtype=torch.int64)
    got = extract_router_expert_ids((weight, idx))
    assert torch.equal(got, idx)


def test_logits_only_uses_topk():
    logits = torch.tensor([[0.1, 5.0, 0.2, 4.0]])
    got = extract_router_expert_ids(logits, top_k=2)
    assert got.tolist() == [[1, 3]]
