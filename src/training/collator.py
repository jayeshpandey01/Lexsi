"""Multi-turn SFT Data Collator with Token-Level Loss Masking (M2.P4).

Implements PyTorch tensor collation for tool-calling agents:
- Prompts (system, human) and tool observations are masked (labels = -100).
- Assistant action tokens (<|START_ACTION|>...<|END_ACTION|>) and responses are supervised (labels = input_ids).
- Fails closed on empty assistant targets (0 active loss tokens).
- Provides deterministic Command-R vocabulary mapping for offline execution.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.data.schemas import LockedTestExportError

logger = logging.getLogger("sft_collator")

# Authoritative Special Tokens for Cohere Command-R / Tiny-Aya
SPECIAL_TOKENS: Dict[str, int] = {
    "<|PAD|>": 0,
    "<|UNK|>": 1,
    "<|BOS|>": 2,
    "<|EOS|>": 3,
    "<|START_OF_TURN_TOKEN|>": 4,
    "<|END_OF_TURN_TOKEN|>": 5,
    "<|USER_TOKEN|>": 6,
    "<|CHATBOT_TOKEN|>": 7,
    "<|SYSTEM_TOKEN|>": 8,
    "<|START_ACTION|>": 9,
    "<|END_ACTION|>": 10,
    "<|START_THINKING|>": 11,
    "<|END_THINKING|>": 12,
}


class DeterministicCommandRTokenizer:
    """Deterministic tokenization engine for Command-R / Tiny-Aya.

    Provides stable integer encoding across both native environments and offline test doubles.
    """

    def __init__(self, vocab_offset: int = 100) -> None:
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.inv_special = {v: k for k, v in self.special_tokens.items()}
        self.vocab_offset = vocab_offset
        self._word_vocab: Dict[str, int] = {}
        self._next_id = vocab_offset
        self.pad_token_id = self.special_tokens["<|PAD|>"]
        self.eos_token_id = self.special_tokens["<|EOS|>"]
        self.bos_token_id = self.special_tokens["<|BOS|>"]

    def _get_word_id(self, word: str) -> int:
        if word not in self._word_vocab:
            self._word_vocab[word] = self._next_id
            self._next_id += 1
        return self._word_vocab[word]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text string into token ID sequence."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        # Tokenize by matching special tokens and whitespace
        words = text.split()
        for w in words:
            if w in self.special_tokens:
                tokens.append(self.special_tokens[w])
            elif "<|START_ACTION|>" in w and "<|END_ACTION|>" in w:
                tokens.append(self.special_tokens["<|START_ACTION|>"])
                inner = w.replace("<|START_ACTION|>", "").replace("<|END_ACTION|>", "")
                for char_chunk in inner.split():
                    tokens.append(self._get_word_id(char_chunk))
                tokens.append(self.special_tokens["<|END_ACTION|>"])
            else:
                tokens.append(self._get_word_id(w))

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        parts: List[str] = []
        for tid in token_ids:
            if tid in self.inv_special:
                parts.append(self.inv_special[tid])
            else:
                # Find matching word in vocab
                matched = [w for w, i in self._word_vocab.items() if i == tid]
                parts.append(matched[0] if matched else f"token_{tid}")
        return " ".join(parts)


class MultiTurnActionDataCollator:
    """Collates ShareGPT conversation records into PyTorch tensors with token-level loss masks.

    Guarantees:
    - Target labels = -100 for all system prompts, user queries, and tool observations.
    - Target labels = input_ids for assistant actions and final response synthesis.
    - Fails closed if any training sequence contains zero supervised assistant tokens.
    - Rejects any conversation contaminated with held-out locked-test task IDs.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer or DeterministicCommandRTokenizer()
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

    def process_single_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Convert a single ShareGPT conversation into input_ids, attention_mask, and labels."""
        input_ids: List[int] = []
        labels: List[int] = []

        # 1. Contamination & fail-closed check
        for turn in conversation:
            val = turn.get("value", "")
            if "task_test_" in val or "manifest_test" in val.lower() or "split: test" in val.lower():
                raise LockedTestExportError(
                    f"Attempted to collate sequestered locked-test conversation: {val[:80]}"
                )

        # 2. Turn-by-turn tokenization and masking
        for turn in conversation:
            role = turn.get("from", "").lower()
            content = turn.get("value", "")

            # Encode turn content
            turn_tokens = self.tokenizer.encode(content)
            if not turn_tokens:
                continue

            input_ids.extend(turn_tokens)

            if role in ("system", "human", "observation", "thought"):
                # Mask out prompts and tool observations
                labels.extend([-100] * len(turn_tokens))
            elif role == "gpt":
                # Supervise assistant actions and synthesized output
                labels.extend(turn_tokens)
            else:
                # Unknown role -> mask out defensively
                labels.extend([-100] * len(turn_tokens))

        # 3. Truncation to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        # 4. Invariant: non-empty assistant target tokens
        supervised_tokens_count = sum(1 for lbl in labels if lbl != -100)
        if supervised_tokens_count == 0:
            raise ValueError(
                "Collated training sequence contains 0 active supervised tokens. "
                "Loss would be undefined (NaN). Training aborted."
            )

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "total_tokens": len(input_ids),
            "supervised_tokens": supervised_tokens_count,
            "masked_tokens": len(input_ids) - supervised_tokens_count,
            "supervision_ratio": round(supervised_tokens_count / len(input_ids), 4),
        }

    def collate_batch(
        self, batch_conversations: List[List[Dict[str, str]]]
    ) -> Dict[str, torch.Tensor]:
        """Collate a batch of conversations into padded PyTorch tensors."""
        processed_samples = [self.process_single_conversation(c) for c in batch_conversations]

        batch_size = len(processed_samples)
        max_batch_len = max(s["total_tokens"] for s in processed_samples)

        if self.pad_to_multiple_of:
            remainder = max_batch_len % self.pad_to_multiple_of
            if remainder != 0:
                max_batch_len += self.pad_to_multiple_of - remainder

        # Tensor allocations
        input_ids_tensor = torch.full(
            (batch_size, max_batch_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask_tensor = torch.zeros(
            (batch_size, max_batch_len), dtype=torch.long
        )
        labels_tensor = torch.full(
            (batch_size, max_batch_len), -100, dtype=torch.long
        )

        for i, s in enumerate(processed_samples):
            length = s["total_tokens"]
            input_ids_tensor[i, :length] = torch.tensor(s["input_ids"], dtype=torch.long)
            attention_mask_tensor[i, :length] = torch.tensor(s["attention_mask"], dtype=torch.long)
            labels_tensor[i, :length] = torch.tensor(s["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
        }

