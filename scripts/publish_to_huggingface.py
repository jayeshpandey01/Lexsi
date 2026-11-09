#!/usr/bin/env python3
"""One-click Hugging Face publisher for Lexsi Track 3 artifacts.

Order of operations:
1. Create Dataset Repo (jayesh20/lexsi-track3-procurement-benchmarks) and upload all files (<1 MB, fast).
2. Create/Update Model Repo (jayesh20/lexsi-track3-aya-lora) with metadata files (README.md, article, configs).
3. Create both Public Collections and link the repos.
4. Upload large model weights (tokenizer.json, adapter_model.safetensors).
5. Output final live URLs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

root_dir = Path(__file__).resolve().parent.parent


def prepare_dataset_folder(staging_dir: Path) -> None:
    """Stage datasets, traces, and manifests into a clean folder for upload."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. SFT datasets
    sft_v2 = root_dir / "data" / "sft" / "sft_train_v2_sharegpt.json"
    if sft_v2.exists():
        shutil.copy2(sft_v2, staging_dir / "sft_train_v2_sharegpt.json")

    sft_v1 = root_dir / "data" / "sft" / "sft_train_v1_sharegpt.json"
    if sft_v1.exists():
        shutil.copy2(sft_v1, staging_dir / "sft_train_v1_sharegpt.json")

    # 2. Manifests (Val and Train only - NEVER Test)
    manifests_dir = staging_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    val_manifest = root_dir / "configs" / "manifests" / "task_manifest_val.json"
    train_manifest = root_dir / "configs" / "manifests" / "task_manifest_train.json"
    if val_manifest.exists():
        shutil.copy2(val_manifest, manifests_dir / "task_manifest_val.json")
    if train_manifest.exists():
        shutil.copy2(train_manifest, manifests_dir / "task_manifest_train.json")

    # 3. Validation Rollout Traces
    traces_dir = staging_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    c3_traces = root_dir / "results" / "runs" / "sft_aya_3b_v2_c3" / "val_rollout_traces.json"
    c2_traces = root_dir / "results" / "runs" / "sft_aya_3b_v2_c2" / "val_rollout_traces.json"
    base_traces = root_dir / "results" / "runs" / "val_base_aya_3b_matched" / "val_rollout_traces.json"
    summary_json = root_dir / "results" / "runs" / "m5_comparison_summary.json"

    if c3_traces.exists():
        shutil.copy2(c3_traces, traces_dir / "val_traces_c3_all_linear.json")
    if c2_traces.exists():
        shutil.copy2(c2_traces, traces_dir / "val_traces_c2_attention_only.json")
    if base_traces.exists():
        shutil.copy2(base_traces, traces_dir / "val_traces_matched_base.json")
    if summary_json.exists():
        shutil.copy2(summary_json, staging_dir / "m5_comparison_summary.json")

    # 4. Copy Data Card as README.md
    data_card = root_dir / "references" / "data_card.md"
    if data_card.exists():
        shutil.copy2(data_card, staging_dir / "README.md")


def prepare_model_metadata_folder(staging_dir: Path) -> None:
    """Stage metadata, configs, and documentation for model repository."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    c3_ckpt = root_dir / "results" / "runs" / "sft_aya_3b_v2_c3" / "checkpoint"
    for fname in ["adapter_config.json", "tokenizer_config.json", "chat_template.jinja"]:
        src = c3_ckpt / fname
        if src.exists():
            shutil.copy2(src, staging_dir / fname)

    model_card = root_dir / "references" / "model_card.md"
    if model_card.exists():
        shutil.copy2(model_card, staging_dir / "README.md")

    article = root_dir / "references" / "huggingface_article.md"
    if article.exists():
        shutil.copy2(article, staging_dir / "research_article.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Lexsi Track 3 artifacts to Hugging Face")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face API token (write access)")
    parser.add_argument("--dry-run", action="store_true", help="Stage files locally without uploading")
    parser.add_argument("--username", help="Hugging Face username (defaults to authenticated user)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: Hugging Face token is required.")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN: Testing staging ===")
        with tempfile.TemporaryDirectory() as tmp_model_dir, tempfile.TemporaryDirectory() as tmp_dataset_dir:
            prepare_model_metadata_folder(Path(tmp_model_dir))
            prepare_dataset_folder(Path(tmp_dataset_dir))
            print(f"Staged {len(list(Path(tmp_model_dir).iterdir()))} model metadata files.")
            print(f"Staged {len(list(Path(tmp_dataset_dir).rglob('*')))} dataset files.")
            return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user_info = api.whoami()
    username = args.username or user_info["name"]
    print(f"Authenticated as Hugging Face user: {username}")

    model_repo_id = f"{username}/lexsi-track3-aya-lora"
    dataset_repo_id = f"{username}/lexsi-track3-procurement-benchmarks"

    # --------------------------------------------------------------------------
    # 1. Dataset Repository Upload
    # --------------------------------------------------------------------------
    print(f"\n[1/5] Creating and uploading Dataset repository: {dataset_repo_id}...")
    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmp_dataset_dir:
        prepare_dataset_folder(Path(tmp_dataset_dir))
        print("Uploading datasets, manifests, traces, and Data Card...")
        api.upload_folder(
            folder_path=tmp_dataset_dir,
            repo_id=dataset_repo_id,
            repo_type="dataset",
            commit_message="Add SFT datasets, execution traces, manifests, and Data Card",
        )
    dataset_url = f"https://huggingface.co/datasets/{dataset_repo_id}"
    print(f"SUCCESS: Dataset published at {dataset_url}")

    # --------------------------------------------------------------------------
    # 2. Model Metadata Upload
    # --------------------------------------------------------------------------
    print(f"\n[2/5] Initializing Model repository & uploading metadata: {model_repo_id}...")
    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmp_model_dir:
        prepare_model_metadata_folder(Path(tmp_model_dir))
        print("Uploading Model Card, research article, and adapter configs...")
        api.upload_folder(
            folder_path=tmp_model_dir,
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Add Model Card, research article, and adapter configs",
        )
    model_url = f"https://huggingface.co/{model_repo_id}"
    print(f"SUCCESS: Model metadata published at {model_url}")

    # --------------------------------------------------------------------------
    # 3. Create Collections
    # --------------------------------------------------------------------------
    print(f"\n[3/5] Creating Hugging Face Model & Dataset Collections...")
    model_collection = api.create_collection(
        title="Lexsi Track 3: Agentic Fine-Tuning Models",
        description="Fine-tuned LoRA adapter checkpoints and matched base comparators for the Lexsi AI Safety & Alignment Hackathon (Track 3: Agentic Fine-Tuning).",
        private=False,
        exists_ok=True,
    )
    api.add_collection_item(
        collection_slug=model_collection.slug,
        item_id=model_repo_id,
        item_type="model",
        note="Fine-tuned Tiny-Aya 3.35B LoRA adapter for procurement tool-calling.",
        exists_ok=True,
    )
    model_col_url = model_collection.url or f"https://huggingface.co/collections/{model_collection.slug}"
    print(f"SUCCESS: Model collection available at {model_col_url}")

    dataset_collection = api.create_collection(
        title="Lexsi Track 3: Agentic Datasets & Traces",
        description="Curated SFT datasets, multi-turn ReAct rollout traces, and evaluation logs for Lexsi Track 3.",
        private=False,
        exists_ok=True,
    )
    api.add_collection_item(
        collection_slug=dataset_collection.slug,
        item_id=dataset_repo_id,
        item_type="dataset",
        note="Curated procurement SFT datasets (v1, v2) and multi-turn execution traces.",
        exists_ok=True,
    )
    dataset_col_url = dataset_collection.url or f"https://huggingface.co/collections/{dataset_collection.slug}"
    print(f"SUCCESS: Dataset collection available at {dataset_col_url}")

    # --------------------------------------------------------------------------
    # 4. Upload Large Model Weights
    # --------------------------------------------------------------------------
    c3_ckpt = root_dir / "results" / "runs" / "sft_aya_3b_v2_c3" / "checkpoint"
    tok_json = c3_ckpt / "tokenizer.json"
    weights_file = c3_ckpt / "adapter_model.safetensors"

    print(f"\n[4/5] Uploading tokenizer.json (21 MB)...")
    if tok_json.exists():
        api.upload_file(
            path_or_fileobj=str(tok_json),
            path_in_repo="tokenizer.json",
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Add tokenizer.json",
        )
        print("SUCCESS: tokenizer.json uploaded.")

    print(f"\n[5/5] Uploading adapter_model.safetensors (59 MB)...")
    if weights_file.exists():
        api.upload_file(
            path_or_fileobj=str(weights_file),
            path_in_repo="adapter_model.safetensors",
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Add adapter_model.safetensors (Condition C3 all-linear LoRA weights)",
        )
        print("SUCCESS: adapter_model.safetensors uploaded.")

    print("\n" + "=" * 78)
    print("ALL ARTIFACTS SUCCESSFULLY PUBLISHED TO HUGGING FACE!")
    print("=" * 78)
    print(f"Hugging Face model collection:\n{model_col_url}\n")
    print(f"Hugging Face dataset collection:\n{dataset_col_url}\n")
    print(f"Hugging Face article:\n{model_url}/blob/main/research_article.md\n")
    print(f"GitHub repo:\nhttps://github.com/jayeshpandey01/Lexsi\n")
    print("=" * 78)


if __name__ == "__main__":
    main()
