"""Unit tests for multi-turn action data collator and token loss masking (M2.P4)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.data.schemas import LockedTestExportError
from src.training.collator import (
    DeterministicCommandRTokenizer,
    MultiTurnActionDataCollator,
)


def test_deterministic_command_r_tokenizer() -> None:
    """Verify tokenizer encodes special Command-R tokens and decodes accurately."""
    tok = DeterministicCommandRTokenizer()
    text = "Hello <|START_ACTION|> lookup_records <|END_ACTION|> world"
    ids = tok.encode(text)

    assert tok.special_tokens["<|START_ACTION|>"] in ids
    assert tok.special_tokens["<|END_ACTION|>"] in ids
    assert len(ids) > 4

    decoded = tok.decode(ids)
    assert "<|START_ACTION|>" in decoded
    assert "<|END_ACTION|>" in decoded


def test_collator_loss_masking_invariants() -> None:
    """Verify that system, human, and observation turns receive -100 while gpt receives input_ids."""
    collator = MultiTurnActionDataCollator()

    sample_conv = [
        {"from": "system", "value": "You are a procurement assistant."},
        {"from": "human", "value": "Calculate total spend for VND-101."},
        {
            "from": "gpt",
            "value": '<|START_ACTION|>{"name": "lookup_records", "arguments": {"vendor": "VND-101"}}<|END_ACTION|>',
        },
        {"from": "observation", "value": '[{"po_id": "PO-100", "total": 500.0}]'},
        {"from": "gpt", "value": "The total approved spend for VND-101 is $500.00."},
    ]

    res = collator.process_single_conversation(sample_conv)
    input_ids = res["input_ids"]
    labels = res["labels"]

    assert len(input_ids) == len(labels)
    assert res["supervised_tokens"] > 0
    assert res["masked_tokens"] > 0
    assert 0.0 < res["supervision_ratio"] < 1.0

    # Verify that wherever labels != -100, labels == input_ids
    for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
        if lbl != -100:
            assert lbl == inp, f"Mismatch at index {i}: label {lbl} != input_id {inp}"


def test_collator_empty_target_fail_closed() -> None:
    """Verify that conversation without any assistant target tokens raises ValueError."""
    collator = MultiTurnActionDataCollator()
    no_assistant_conv = [
        {"from": "system", "value": "System prompt only."},
        {"from": "human", "value": "User prompt only."},
    ]
    with pytest.raises(ValueError, match="0 active supervised tokens"):
        collator.process_single_conversation(no_assistant_conv)


def test_collator_locked_test_sequestration_guard() -> None:
    """Verify that conversation referencing locked test tasks raises LockedTestExportError."""
    collator = MultiTurnActionDataCollator()
    contaminated_conv = [
        {"from": "system", "value": "System"},
        {"from": "human", "value": "Audit task_test_T3_vendor_rating_check_001 for vendor VND-108"},
        {"from": "gpt", "value": "Output"},
    ]
    with pytest.raises(LockedTestExportError, match="locked-test"):
        collator.process_single_conversation(contaminated_conv)


def test_collator_batch_tensor_collation() -> None:
    """Verify collate_batch returns valid PyTorch tensors with correct shapes and padding."""
    collator = MultiTurnActionDataCollator()

    conv1 = [
        {"from": "system", "value": "Short sys."},
        {"from": "human", "value": "Short query."},
        {"from": "gpt", "value": "Short answer."},
    ]
    conv2 = [
        {"from": "system", "value": "A much longer system prompt with extra explanation."},
        {"from": "human", "value": "A longer query inquiring about multiple vendor POs."},
        {
            "from": "gpt",
            "value": '<|START_ACTION|>{"name": "lookup_records", "arguments": {}}<|END_ACTION|>',
        },
        {"from": "observation", "value": "[1, 2, 3, 4, 5]"},
        {"from": "gpt", "value": "Detailed final synthesis text."},
    ]

    batch = collator.collate_batch([conv1, conv2])

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    assert input_ids.shape == (2, max(len(collator.tokenizer.encode(" ".join(t["value"] for t in c))) for c in [conv1, conv2]))
    assert attention_mask.shape == input_ids.shape
    assert labels.shape == input_ids.shape

    # Padding in conv1 should be pad_token_id (0) and label -100
    len1 = collator.process_single_conversation(conv1)["total_tokens"]
    assert torch.all(input_ids[0, len1:] == collator.pad_token_id)
    assert torch.all(attention_mask[0, len1:] == 0)
    assert torch.all(labels[0, len1:] == -100)


def test_collator_on_real_training_corpus() -> None:
    """Verify all 20 conversations in data/sft/sft_train_sharegpt.json process with valid non-zero loss masks."""
    train_sg_path = Path("data/sft/sft_train_sharegpt.json")
    assert train_sg_path.exists()

    conversations = json.loads(train_sg_path.read_text(encoding="utf-8"))
    assert len(conversations) == 20

    collator = MultiTurnActionDataCollator()

    for idx, conv in enumerate(conversations):
        res = collator.process_single_conversation(conv)
        assert res["supervised_tokens"] > 0, f"Example {idx} has 0 target tokens"
        assert res["masked_tokens"] > 0
        assert res["total_tokens"] > 0
        assert 0.05 < res["supervision_ratio"] < 0.50

    # Verify batch collation over full dataset in mini-batches
    batch_4 = collator.collate_batch(conversations[:4])
    assert batch_4["input_ids"].shape[0] == 4
    assert torch.sum(batch_4["labels"] != -100) > 0

