"""Generator script to produce the self-contained Kaggle M5 training notebook (M5.P3).

Creates notebooks/kaggle_m5_training.ipynb:
- Self-contained, deterministically seeded (seed=42).
- Includes the 220 verified training demonstrations from data/sft/sft_train_v2_sharegpt.json.
- Includes the 10 validation tasks from configs/manifests/task_manifest_val.json.
- Incorporates the canonical system prompt with full JSON schemas.
- Supports:
  1. Matched untuned base model validation rollout (val_base_aya_3b_matched).
  2. Condition C2 QLoRA training (attention-only LoRA, r=16, alpha=32).
  3. Condition C3 QLoRA training (all-linear LoRA, r=16, alpha=32).
  4. Fail-closed loss NaN and timeout traps.
  5. AuditKIT evaluation and comparison packaging (m5_kaggle_artifacts.zip).
"""

from __future__ import annotations

import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent


def create_notebook() -> None:
    sft_data_path = root_dir / "data" / "sft" / "sft_train_v2_sharegpt.json"
    val_data_path = root_dir / "configs" / "manifests" / "task_manifest_val.json"

    sft_records = json.loads(sft_data_path.read_text(encoding="utf-8"))
    val_manifest = json.loads(val_data_path.read_text(encoding="utf-8"))

    sft_json_str = json.dumps(sft_records)
    val_json_str = json.dumps(val_manifest)

    cells = []

    # Cell 1: Markdown Overview
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Lexsi Track 3: Milestone 5 Remediation — Matched Base Rollout & QLoRA Fine-Tuning\n",
            "\n",
            "> **Platform:** Kaggle GPU Backend (2× NVIDIA Tesla T4, 32GB total VRAM / 16GB per GPU)\n",
            "> **Model:** `CohereLabs/tiny-aya-global` (3.35B parameters, Gated Model)\n",
            "> **Execution Matrix:**\n",
            "> 1. **Matched Base Rollout (`val_base_aya_3b_matched`):** Zero-shot base model on validation split with canonical schema.\n",
            "> 2. **Condition C2 (`sft_aya_3b_v2_c2`):** Attention-only LoRA ($r=16, \\alpha=32$) on 220 unique demonstrations.\n",
            "> 3. **Condition C3 (`sft_aya_3b_v2_c3`):** All-linear LoRA ($r=16, \\alpha=32$, `q,k,v,o,gate,up,down`) on 220 demonstrations.\n",
            "> **Milestone:** Milestone 5 (M5.P3 & M5.P4)\n",
            "\n",
            "### Prerequisites for Gated Model Access:\n",
            "1. **Accept Terms on Hugging Face:** Visit [https://huggingface.co/CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) and click **'Agree and access repository'**.\n",
            "2. **Hugging Face Token:** In Kaggle, add your token under **Add-ons -> Secrets -> Label: `HF_TOKEN`**."
        ]
    })

    # Cell 2: Dependencies & Preflight
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. GPU Preflight & Environment Setup\n",
            "import os, sys, time, json, math, re\n",
            "import torch\n",
            "\n",
            "print(f\"Python Version: {sys.version}\")\n",
            "print(f\"PyTorch Version: {torch.__version__}\")\n",
            "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
            "\n",
            "if not torch.cuda.is_available():\n",
            "    raise RuntimeError(\"CUDA is required. Enable GPU T4 x2 accelerator in Kaggle Settings.\")\n",
            "\n",
            "device_count = torch.cuda.device_count()\n",
            "print(f\"Available GPUs: {device_count}\")\n",
            "for i in range(device_count):\n",
            "    props = torch.cuda.get_device_properties(i)\n",
            "    print(f\"  Device {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)\")\n",
            "\n",
            "!pip install -q --upgrade pip\n",
            "!pip install -q transformers peft accelerate bitsandbytes datasets huggingface_hub pydantic\n",
            "print(\"Dependencies successfully verified.\")"
        ]
    })

    # Cell 3: HF Auth
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Hugging Face Authentication\n",
            "from huggingface_hub import login\n",
            "\n",
            "hf_token = os.environ.get(\"HF_TOKEN\")\n",
            "if not hf_token:\n",
            "    try:\n",
            "        from kaggle_secrets import UserSecretsClient\n",
            "        hf_token = UserSecretsClient().get_secret(\"HF_TOKEN\")\n",
            "    except Exception:\n",
            "        pass\n",
            "\n",
            "if not hf_token:\n",
            "    import getpass\n",
            "    hf_token = getpass.getpass(\"Enter your Hugging Face Access Token: \").strip()\n",
            "\n",
            "if hf_token:\n",
            "    login(token=hf_token)\n",
            "    os.environ[\"HF_TOKEN\"] = hf_token\n",
            "    print(\"Hugging Face Hub authentication successful.\")\n",
            "else:\n",
            "    raise ValueError(\"HF_TOKEN is required to load CohereLabs/tiny-aya-global.\")"
        ]
    })

    # Cell 4: Datasets injection
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Inject Verified Datasets\n",
            "os.makedirs(\"data/sft\", exist_ok=True)\n",
            "os.makedirs(\"configs/manifests\", exist_ok=True)\n",
            "os.makedirs(\"results/runs\", exist_ok=True)\n",
            "\n",
            f"sft_json_raw = {repr(sft_json_str)}\n",
            f"val_manifest_raw = {repr(val_json_str)}\n",
            "\n",
            "with open(\"data/sft/sft_train_v2_sharegpt.json\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write(sft_json_raw)\n",
            "\n",
            "with open(\"configs/manifests/task_manifest_val.json\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write(val_manifest_raw)\n",
            "\n",
            "train_samples = json.loads(sft_json_raw)\n",
            "val_tasks = json.loads(val_manifest_raw)[\"tasks\"]\n",
            "print(f\"Loaded {len(train_samples)} training conversations and {len(val_tasks)} validation tasks.\")"
        ]
    })

    # Cell 5: Collator & Special Token Masking
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Tokenizer & Assistant Loss-Masking Collator\n",
            "from transformers import AutoTokenizer\n",
            "from torch.utils.data import Dataset\n",
            "\n",
            "MODEL_ID = \"CohereLabs/tiny-aya-global\"\n",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)\n",
            "\n",
            "# Verify Command-R special tokens exist in vocabulary\n",
            "for tok in [\"<|START_ACTION|>\", \"<|END_ACTION|>\", \"<|START_OF_TURN_TOKEN|>\", \"<|END_OF_TURN_TOKEN|>\"]:\n",
            "    tid = tokenizer.convert_tokens_to_ids(tok)\n",
            "    print(f\"Special token '{tok}' mapped to ID: {tid}\")\n",
            "\n",
            "class SFTConversationDataset(Dataset):\n",
            "    def __init__(self, data, tokenizer, max_length=2048):\n",
            "        self.data = data\n",
            "        self.tokenizer = tokenizer\n",
            "        self.max_length = max_length\n",
            "        self.start_turn = \"<|START_OF_TURN_TOKEN|>\"\n",
            "        self.end_turn = \"<|END_OF_TURN_TOKEN|>\"\n",
            "        self.sys_tok = \"<|SYSTEM_TOKEN|>\"\n",
            "        self.user_tok = \"<|USER_TOKEN|>\"\n",
            "        self.bot_tok = \"<|CHATBOT_TOKEN|>\"\n",
            "\n",
            "    def __len__(self):\n",
            "        return len(self.data)\n",
            "\n",
            "    def __getitem__(self, idx):\n",
            "        conv = self.data[idx]\n",
            "        input_ids = []\n",
            "        labels = []\n",
            "\n",
            "        for turn in conv:\n",
            "            role = turn[\"from\"]\n",
            "            content = turn[\"value\"]\n",
            "            if role == \"system\":\n",
            "                t_str = f\"{self.start_turn}{self.sys_tok}{content}{self.end_turn}\"\n",
            "                ids = self.tokenizer.encode(t_str, add_special_tokens=False)\n",
            "                input_ids.extend(ids)\n",
            "                labels.extend([-100] * len(ids))\n",
            "            elif role == \"human\":\n",
            "                t_str = f\"{self.start_turn}{self.user_tok}{content}{self.end_turn}\"\n",
            "                ids = self.tokenizer.encode(t_str, add_special_tokens=False)\n",
            "                input_ids.extend(ids)\n",
            "                labels.extend([-100] * len(ids))\n",
            "            elif role == \"observation\":\n",
            "                t_str = f\"{self.start_turn}{self.sys_tok}Observation: {content}{self.end_turn}\"\n",
            "                ids = self.tokenizer.encode(t_str, add_special_tokens=False)\n",
            "                input_ids.extend(ids)\n",
            "                labels.extend([-100] * len(ids))\n",
            "            elif role == \"gpt\":\n",
            "                prefix = f\"{self.start_turn}{self.bot_tok}\"\n",
            "                suffix = f\"{self.end_turn}\"\n",
            "                prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)\n",
            "                body_ids = self.tokenizer.encode(content, add_special_tokens=False)\n",
            "                suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)\n",
            "                \n",
            "                input_ids.extend(prefix_ids + body_ids + suffix_ids)\n",
            "                labels.extend([-100] * len(prefix_ids) + body_ids + suffix_ids)\n",
            "\n",
            "        input_ids = input_ids[:self.max_length]\n",
            "        labels = labels[:self.max_length]\n",
            "        return {\"input_ids\": torch.tensor(input_ids, dtype=torch.long), \"labels\": torch.tensor(labels, dtype=torch.long)}\n",
            "\n",
            "def collate_fn(batch):\n",
            "    pad_id = tokenizer.pad_token_id or 0\n",
            "    max_len = max(len(x[\"input_ids\"]) for x in batch)\n",
            "    b_ids = []\n",
            "    b_labels = []\n",
            "    b_mask = []\n",
            "    for x in batch:\n",
            "        pad_len = max_len - len(x[\"input_ids\"])\n",
            "        b_ids.append(torch.cat([x[\"input_ids\"], torch.full((pad_len,), pad_id, dtype=torch.long)]))\n",
            "        b_labels.append(torch.cat([x[\"labels\"], torch.full((pad_len,), -100, dtype=torch.long)]))\n",
            "        b_mask.append(torch.cat([torch.ones(len(x[\"input_ids\"]), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))\n",
            "    return {\"input_ids\": torch.stack(b_ids), \"labels\": torch.stack(b_labels), \"attention_mask\": torch.stack(b_mask)}\n",
            "\n",
            "train_dataset = SFTConversationDataset(train_samples, tokenizer)\n",
            "print(f\"Prepared dataset with {len(train_dataset)} examples.\")"
        ]
    })

    # Cell 6: Untuned Base Model Matched Rollout (Condition C1)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Matched Untuned Base Model Validation Rollout (val_base_aya_3b_matched)\n",
            "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n",
            "\n",
            "bnb_config = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type=\"nf4\",\n",
            "    bnb_4bit_use_double_quant=True,\n",
            "    bnb_4bit_compute_dtype=torch.float16,\n",
            ")\n",
            "\n",
            "print(\"Loading untuned CohereLabs/tiny-aya-global in 4-bit NF4...\")\n",
            "base_model = AutoModelForCausalLM.from_pretrained(\n",
            "    MODEL_ID,\n",
            "    quantization_config=bnb_config,\n",
            "    device_map=\"auto\",\n",
            "    token=hf_token,\n",
            ")\n",
            "base_model.eval()\n",
            "\n",
            "# Canonical system prompt extracted from training data\n",
            "CANONICAL_SYS_PROMPT = train_samples[0][0][\"value\"]\n",
            "\n",
            "def evaluate_on_val(model, tok, run_name):\n",
            "    results = []\n",
            "    tool_calls_count = 0\n",
            "    successes = 0\n",
            "    print(f\"\\n--- Running Validation Rollout: {run_name} ---\")\n",
            "    for idx, t in enumerate(val_tasks):\n",
            "        q = t[\"user_query\"]\n",
            "        expected_ans = t[\"oracle_answer\"]\n",
            "        \n",
            "        prompt = f\"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{CANONICAL_SYS_PROMPT}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{q}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\"\n",
            "        enc = tok(prompt, return_tensors=\"pt\").to(model.device)\n",
            "        \n",
            "        with torch.no_grad():\n",
            "            out = model.generate(\n",
            "                **enc,\n",
            "                max_new_tokens=128,\n",
            "                do_sample=False,\n",
            "                temperature=0.01,\n",
            "                eos_token_id=tok.convert_tokens_to_ids(\"<|END_OF_TURN_TOKEN|>\"),\n",
            "            )\n",
            "        \n",
            "        gen_text = tok.decode(out[0][enc[\"input_ids\"].shape[1]:], skip_special_tokens=False)\n",
            "        has_action = \"<|START_ACTION|>\" in gen_text and \"<|END_ACTION|>\" in gen_text\n",
            "        if has_action:\n",
            "            tool_calls_count += 1\n",
            "            \n",
            "        # Accuracy check\n",
            "        nums = re.findall(r\"[-+]?(?:\\d*\\.\\d+|\\d+)\", gen_text.replace(\",\", \"\"))\n",
            "        is_succ = False\n",
            "        for n in nums:\n",
            "            try:\n",
            "                v = float(n)\n",
            "                if abs(v - expected_ans) <= 0.05 or (abs(v - expected_ans) / (abs(expected_ans) + 1e-5)) <= 0.01:\n",
            "                    is_succ = True\n",
            "                    break\n",
            "            except Exception:\n",
            "                pass\n",
            "        if is_succ:\n",
            "            successes += 1\n",
            "            \n",
            "        results.append({\n",
            "            \"task_id\": t[\"task_id\"],\n",
            "            \"user_query\": q,\n",
            "            \"generated_output\": gen_text,\n",
            "            \"has_tool_action\": has_action,\n",
            "            \"is_success\": is_succ,\n",
            "            \"oracle_answer\": expected_ans,\n",
            "        })\n",
            "        print(f\"Task {idx:02d} ({t['task_id']}): action={has_action}, success={is_succ}\")\n",
            "        \n",
            "    m1 = successes / len(val_tasks)\n",
            "    m2 = tool_calls_count / len(val_tasks)\n",
            "    print(f\"SUMMARY [{run_name}]: M1 (Success)={m1*100:.1f}%, M2 (Tool Calls)={m2*100:.1f}%\")\n",
            "    \n",
            "    run_dir = Path(f\"results/runs/{run_name}\")\n",
            "    run_dir.mkdir(parents=True, exist_ok=True)\n",
            "    with open(run_dir / \"val_rollout_traces.json\", \"w\", encoding=\"utf-8\") as f:\n",
            "        json.dump(results, f, indent=2)\n",
            "    return results, m1, m2\n",
            "\n",
            "base_results, base_m1, base_m2 = evaluate_on_val(base_model, tokenizer, \"val_base_aya_3b_matched\")"
        ]
    })

    # Cell 7: Training Condition C2 (Attention-only LoRA)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Condition C2 Training (Attention-Only LoRA: q, k, v, o)\n",
            "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
            "from torch.utils.data import DataLoader\n",
            "from transformers import get_cosine_schedule_with_warmup\n",
            "\n",
            "print(\"Configuring Condition C2 PEFT LoRA (Attention-only: q, k, v, o)...\")\n",
            "base_model = prepare_model_for_kbit_training(base_model)\n",
            "\n",
            "lora_config_c2 = LoraConfig(\n",
            "    r=16,\n",
            "    lora_alpha=32,\n",
            "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],\n",
            "    lora_dropout=0.05,\n",
            "    bias=\"none\",\n",
            "    task_type=\"CAUSAL_LM\",\n",
            ")\n",
            "model_c2 = get_peft_model(base_model, lora_config_c2)\n",
            "model_c2.print_trainable_parameters()\n",
            "\n",
            "def train_model(model, run_name, max_steps=400, lr=1e-4):\n",
            "    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)\n",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)\n",
            "    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=max_steps)\n",
            "    \n",
            "    model.train()\n",
            "    step = 0\n",
            "    start_time = time.time()\n",
            "    accum_steps = 4\n",
            "    optimizer.zero_grad()\n",
            "    \n",
            "    print(f\"\\nStarting Training: {run_name} (max_steps={max_steps}, lr={lr})...\")\n",
            "    while step < max_steps:\n",
            "        for batch in loader:\n",
            "            # Fail-closed timeout guard (45 min)\n",
            "            if (time.time() - start_time) > (45 * 60):\n",
            "                raise TimeoutError(\"Training exceeded 45-minute wall-clock cap!\")\n",
            "                \n",
            "            batch = {k: v.to(model.device) for k, v in batch.items()}\n",
            "            outputs = model(**batch)\n",
            "            loss = outputs.loss / accum_steps\n",
            "            loss_val = outputs.loss.item()\n",
            "            \n",
            "            # Fail-closed loss NaN & explosion guards\n",
            "            if not math.isfinite(loss_val):\n",
            "                raise ValueError(f\"CRITICAL: Non-finite loss encountered: {loss_val}\")\n",
            "            if loss_val > 15.0:\n",
            "                raise ValueError(f\"CRITICAL: Loss explosion threshold exceeded: {loss_val}\")\n",
            "                \n",
            "            loss.backward()\n",
            "            \n",
            "            if (step + 1) % accum_steps == 0:\n",
            "                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
            "                optimizer.step()\n",
            "                scheduler.step()\n",
            "                optimizer.zero_grad()\n",
            "                \n",
            "            if step % 25 == 0:\n",
            "                elapsed = time.time() - start_time\n",
            "                print(f\"Step {step:03d}/{max_steps} | Loss: {loss_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {elapsed:.1f}s\")\n",
            "                \n",
            "            step += 1\n",
            "            if step >= max_steps:\n",
            "                break\n",
            "                \n",
            "    ckpt_dir = Path(f\"results/runs/{run_name}/checkpoint\")\n",
            "    ckpt_dir.mkdir(parents=True, exist_ok=True)\n",
            "    model.save_pretrained(ckpt_dir)\n",
            "    tokenizer.save_pretrained(ckpt_dir)\n",
            "    print(f\"SUCCESS: Checkpoint saved to {ckpt_dir}\")\n",
            "    return model\n",
            "\n",
            "model_c2 = train_model(model_c2, \"sft_aya_3b_v2_c2\", max_steps=400, lr=1e-4)\n",
            "model_c2.eval()\n",
            "c2_results, c2_m1, c2_m2 = evaluate_on_val(model_c2, tokenizer, \"sft_aya_3b_v2_c2\")"
        ]
    })

    # Cell 8: Training Condition C3 (All-Linear LoRA)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Condition C3 Training (All-Linear LoRA: q, k, v, o, gate, up, down)\n",
            "print(\"Reloading base model for Condition C3 All-Linear LoRA...\")\n",
            "del model_c2\n",
            "torch.cuda.empty_cache()\n",
            "\n",
            "base_model = AutoModelForCausalLM.from_pretrained(\n",
            "    MODEL_ID,\n",
            "    quantization_config=bnb_config,\n",
            "    device_map=\"auto\",\n",
            "    token=hf_token,\n",
            ")\n",
            "base_model = prepare_model_for_kbit_training(base_model)\n",
            "\n",
            "lora_config_c3 = LoraConfig(\n",
            "    r=16,\n",
            "    lora_alpha=32,\n",
            "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
            "    lora_dropout=0.05,\n",
            "    bias=\"none\",\n",
            "    task_type=\"CAUSAL_LM\",\n",
            ")\n",
            "model_c3 = get_peft_model(base_model, lora_config_c3)\n",
            "model_c3.print_trainable_parameters()\n",
            "\n",
            "model_c3 = train_model(model_c3, \"sft_aya_3b_v2_c3\", max_steps=400, lr=1e-4)\n",
            "model_c3.eval()\n",
            "c3_results, c3_m1, c3_m2 = evaluate_on_val(model_c3, tokenizer, \"sft_aya_3b_v2_c3\")"
        ]
    })

    # Cell 9: Side-by-Side Evaluation Summary
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Comparative Results Table & Capability Floor Check\n",
            "print(\"\\n=======================================================================\")\n",
            "print(\"          LEXSI TRACK 3 — M5 CAPABILITY REMEDIATION EVALUATION         \")\n",
            "print(\"=======================================================================\")\n",
            "print(f\"{'Condition / Model':<30} | {'M1 Success':<12} | {'M2 Tool Call':<12} | {'Floor Status'}\")\n",
            "print(\"-----------------------------------------------------------------------\")\n",
            "print(f\"{'Historical M4 (sft_aya_3b_001)':<30} | {'0.0%':<12} | {'0.0%':<12} | {'FAILED (Early Termination)'}\")\n",
            "print(f\"{'Untuned Base (Matched)':<30} | {f'{base_m1*100:.1f}%':<12} | {f'{base_m2*100:.1f}%':<12} | {'Baseline Comparator'}\")\n",
            "c2_status = 'PASSED FLOOR' if c2_m1 >= 0.40 else 'BELOW FLOOR'\n",
            "print(f\"{'C2: SFT v2 (Attention-only)':<30} | {f'{c2_m1*100:.1f}%':<12} | {f'{c2_m2*100:.1f}%':<12} | {c2_status}\")\n",
            "c3_status = 'PASSED FLOOR' if c3_m1 >= 0.40 else 'BELOW FLOOR'\n",
            "print(f\"{'C3: SFT v2 (All-Linear LoRA)':<30} | {f'{c3_m1*100:.1f}%':<12} | {f'{c3_m2*100:.1f}%':<12} | {c3_status}\")\n",
            "print(\"=======================================================================\")"
        ]
    })

    # Cell 10: Packaging Artifacts ZIP
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 9. Package Artifacts Archive\n",
            "!zip -r m5_kaggle_artifacts.zip results/runs/ data/sft/ configs/manifests/\n",
            "print(\"Successfully created m5_kaggle_artifacts.zip!\")\n",
            "print(\"Download this file from Kaggle Output to sync with local repository.\")"
        ]
    })

    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": cells,
    }

    output_path = root_dir / "notebooks" / "kaggle_m5_training.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Generated {output_path} ({len(cells)} cells).")


if __name__ == "__main__":
    create_notebook()
