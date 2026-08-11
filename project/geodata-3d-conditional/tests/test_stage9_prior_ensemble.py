from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.generator_posterior import projected_fixed_euler_prior_sample
from guidance.prior_ensemble import (
    candidate_id,
    file_sha256,
    gaussian_source,
    generate_prior_batch,
    hard_seismic_response,
    load_tensor_gzip,
    save_tensor_gzip,
    source_seed,
    validate_protocol_config,
)
from guidance.seismic import tensor_sha256
from scripts.stage9.common import create_staging_directory


class _ConstantVelocityNet(torch.nn.Module):
    def forward(self, state, conditioning, time):
        del conditioning, time
        return torch.full_like(state, 0.04)


class _DummyModel:
    def __init__(self):
        self.embedding = torch.nn.Embedding(3, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(3))
        self.net = _ConstantVelocityNet()

    def embed(self, labels):
        value = self.embedding(labels.squeeze(1).long() + 1)
        return value.permute(0, 4, 1, 2, 3).contiguous()

    def decode(self, state):
        state = F.normalize(state, dim=1)
        embeddings = F.normalize(self.embedding.weight, dim=1)
        return torch.einsum("bexyz,ce->bcxyz", state, embeddings).argmax(dim=1)


def _config() -> dict[str, object]:
    path = (
        PROJECT_DIR
        / "experiments/stage9_flow_prior_posterior/configs"
        / "stage9a_prior_support_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_stage9a_config_and_seed_schedule():
    config = _config()
    resolved = validate_protocol_config(config)
    assert resolved["formal_count"] == 1024
    assert resolved["n_steps"] == 32
    assert candidate_id(0) == "candidate_000000"
    assert candidate_id(1023) == "candidate_001023"
    assert source_seed(config, case_index=0, candidate_index=0, mode="formal") == 9301000
    assert source_seed(config, case_index=2, candidate_index=1023, mode="formal") == 9322023
    assert source_seed(config, case_index=0, candidate_index=0, mode="smoke") == 9801000
    assert [case["native_seed"] for case in config["primary_cases"]] == [
        20260901,
        20260902,
        20260903,
    ]


def test_candidate_sources_are_deterministic_and_order_independent():
    left = gaussian_source((1, 3, 2, 2, 2), seed=901)
    unrelated = gaussian_source((1, 3, 2, 2, 2), seed=999)
    right = gaussian_source((1, 3, 2, 2, 2), seed=901)
    assert torch.equal(left, right)
    assert tensor_sha256(left) == tensor_sha256(right)
    assert not torch.equal(left, unrelated)


def test_stage9_projected_sampler_matches_existing_alpha_zero_path():
    torch.manual_seed(5)
    model = _DummyModel()
    conditions = torch.tensor([[[[[-1, 0, 1, 0]]]]], dtype=torch.long)
    mask = torch.zeros_like(conditions, dtype=torch.bool)
    mask[..., 0] = True
    mask[..., 2] = True
    embedded = model.embed(conditions)
    conditioning = torch.where(mask.expand_as(embedded), embedded, torch.zeros_like(embedded))
    initial = torch.randn(2, 3, 1, 1, 4)
    stage9 = generate_prior_batch(
        model,
        initial,
        conditioning,
        embedded,
        mask,
        conditions,
        n_steps=32,
    )
    reference_state = projected_fixed_euler_prior_sample(
        model, initial, conditioning, embedded, mask, n_steps=32
    )
    reference = (model.decode(reference_state) - 1).unsqueeze(1)
    reference[:, :, :, :, 0] = -1
    reference[:, :, :, :, 2] = 1
    assert torch.equal(stage9, reference)
    assert int(((stage9 != conditions) & mask).sum()) == 0


def test_lossless_gzip_cache_preserves_dtype_shape_bytes_and_hash(tmp_path: Path):
    value = torch.randn(4, 1, 4, 5, 6, dtype=torch.float32)
    record = save_tensor_gzip(tmp_path / "chunk.pt.gz", value)
    loaded = load_tensor_gzip(tmp_path / "chunk.pt.gz", expected=record)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, value)
    assert tensor_sha256(loaded) == record["tensor_sha256"]
    assert file_sha256(tmp_path / "chunk.pt.gz") == record["file_sha256"]


def test_hard_seismic_response_reuses_hard_mapping_and_mask():
    labels = torch.tensor([[[[[-1, 0, 1]]]]], dtype=torch.long)
    table = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1
    mask = labels != -1

    def forward(impedance, slowness, subsurface):
        return (impedance + 2 * slowness) * subsurface

    response = hard_seismic_response(
        labels,
        property_table=table,
        subsurface_mask=mask,
        forward_operator=forward,
    )
    assert response.dtype == torch.float32
    assert response[0, 0, 0, 0, 0] == 0
    assert response[0, 0, 0, 0, 1] == table[0, 1] + 2 * table[1, 1]


def test_immutable_output_refusal(tmp_path: Path):
    final = tmp_path / "pool"
    final.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        create_staging_directory(final)
