#!/usr/bin/env python3
"""Builds the production-ready Kaggle notebook for Milestone 5 (M5).

Generates notebooks/kaggle_m5_training.ipynb with full data embedding,
memory isolation between runs, bfloat16 stability, and Kaggle 2xT4 GPU mapping.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Read the SFT training data and Val manifest directly from local files
sft_file = ROOT_DIR / "data" / "sft" / "sft_train_v2_sharegpt.json"
val_file = ROOT_DIR / "configs" / "manifests" / "task_manifest_val.json"

if not sft_file.exists():
    raise FileNotFoundError(f"Missing SFT data: {sft_file}")
if not val_file.exists():
    raise FileNotFoundError(f"Missing Val manifest: {val_file}")

sft_parsed = json.loads(sft_file.read_text(encoding="utf-8"))
val_parsed = json.loads(val_file.read_text(encoding="utf-8"))
assert len(sft_parsed) == 220, f"Expected 220 SFT samples, got {len(sft_parsed)}"
assert len(val_parsed["tasks"]) == 10, f"Expected 10 val tasks, got {len(val_parsed['tasks'])}"

# Compact single-line JSON representations
sft_compact = json.dumps(sft_parsed, separators=(',', ':'))
val_compact = json.dumps(val_parsed, separators=(',', ':'))

cells = []

# ==============================================================================
# Cell 0: Markdown Header & Instructions
# ==============================================================================
cell_0 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Lexsi Track 3: Milestone 5 Remediation — Matched Base Rollout & QLoRA Fine-Tuning\n",
        "\n",
        "> **Platform:** Kaggle GPU Backend (2× NVIDIA Tesla T4, 32GB total VRAM / 16GB per GPU)\n",
        "> **Model:** `CohereLabs/tiny-aya-global` (3.35B parameters, Gated Model)\n",
        "> **Execution Matrix:**\n",
        "> 1. **Matched Base Comparator (`val_base_aya_3b_matched`):** Zero-shot base model on validation split (GPU 0).\n",
        "> 2. **Condition C2 (`sft_aya_3b_v2_c2`):** Attention-only LoRA ($r=16, \\alpha=32$, `q,k,v,o`) on 220 demonstrations (GPU 0).\n",
        "> 3. **Condition C3 (`sft_aya_3b_v2_c3`):** All-linear LoRA ($r=16, \\alpha=32$, `q,k,v,o,gate,up,down`) on 220 demonstrations (GPU 1).\n",
        "> 4. **Comparative Analysis & Floor Check:** Standardized M1-M3 evaluation table with capability floors.\n",
        "> 5. **Artifact Packaging:** Produces `m5_kaggle_artifacts.zip` containing all traces, checkpoints, and configs.\n",
        "\n",
        "### Prerequisites for Gated Model Access:\n",
        "1. **Accept Terms on Hugging Face:** Visit [https://huggingface.co/CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) and click **'Agree and access repository'**.\n",
        "2. **Hugging Face Token:** In Kaggle, add your token under **Add-ons -> Secrets -> Label: `HF_TOKEN`** (or paste when prompted)."
    ]
}
cells.append(cell_0)

# ==============================================================================
# Cell 1: Environment Setup & Preflight
# ==============================================================================
cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. GPU Preflight & Environment Setup\n",
        "import os, sys, time, json, math, re, gc, shutil, hashlib\n",
        "from pathlib import Path\n",
        "import torch\n",
        "\n",
        "print(f\"Python Version: {sys.version}\")\n",
        "print(f\"PyTorch Version: {torch.__version__}\")\n",
        "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
        "\n",
        "if not torch.cuda.is_available():\n",
        "    raise RuntimeError(\"CUDA is required. Enable GPU T4 x2 accelerator in Kaggle Notebook Settings.\")\n",
        "\n",
        "device_count = torch.cuda.device_count()\n",
        "print(f\"Available GPUs: {device_count}\")\n",
        "for i in range(device_count):\n",
        "    props = torch.cuda.get_device_properties(i)\n",
        "    free_mem = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)\n",
        "    print(f\"  Device {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB total, ~{free_mem:.2f} GB free)\")\n",
        "\n",
        "# Optimization flags\n",
        "torch.backends.cuda.matmul.allow_tf32 = True\n",
        "\n",
        "!pip install -q --upgrade pip\n",
        "!pip install -q transformers peft accelerate bitsandbytes datasets huggingface_hub pydantic sentencepiece protobuf\n",
        "print(\"Dependencies successfully installed and verified.\")"
    ]
}
cells.append(cell_1)

# ==============================================================================
# Cell 2: Hugging Face Authentication
# ==============================================================================
cell_2 = {
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
}
cells.append(cell_2)

# ==============================================================================
# Cell 3: Data Injection
# ==============================================================================
cell_3_lines = [
    "# 3. Inject Verified Datasets\n",
    "os.makedirs(\"data/sft\", exist_ok=True)\n",
    "os.makedirs(\"configs/manifests\", exist_ok=True)\n",
    "os.makedirs(\"results/runs\", exist_ok=True)\n",
    "\n",
    f"sft_json_raw = {repr(sft_compact)}\n",
    f"val_manifest_raw = {repr(val_compact)}\n",
    "\n",
    "with open(\"data/sft/sft_train_v2_sharegpt.json\", \"w\", encoding=\"utf-8\") as f:\n",
    "    f.write(sft_json_raw)\n",
    "\n",
    "with open(\"configs/manifests/task_manifest_val.json\", \"w\", encoding=\"utf-8\") as f:\n",
    "    f.write(val_manifest_raw)\n",
    "\n",
    "train_samples = json.loads(sft_json_raw)\n",
    "val_tasks = json.loads(val_manifest_raw)[\"tasks\"]\n",
    "assert len(train_samples) == 220, f\"Expected 220 training samples, got {len(train_samples)}\"\n",
    "assert len(val_tasks) == 10, f\"Expected 10 validation tasks, got {len(val_tasks)}\"\n",
    "print(f\"Loaded {len(train_samples)} training conversations and {len(val_tasks)} validation tasks successfully.\")"
]

cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell_3_lines
}
cells.append(cell_3)

# ==============================================================================
# Cell 4: Tokenizer, Memory Isolation & Common Pipeline Functions
# ==============================================================================
cell_4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 4. Tokenizer, Memory Isolation & Common Pipeline Functions\n",
        "from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, AutoModelForCausalLM, BitsAndBytesConfig\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
        "\n",
        "MODEL_ID = \"CohereLabs/tiny-aya-global\"\n",
        "print(f\"Loading tokenizer for {MODEL_ID}...\")\n",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)\n",
        "if tokenizer.pad_token_id is None:\n",
        "    tokenizer.pad_token_id = tokenizer.eos_token_id\n",
        "\n",
        "# Verify Command-R special tokens exist in vocabulary\n",
        "for tok in [\"<|START_ACTION|>\", \"<|END_ACTION|>\", \"<|START_OF_TURN_TOKEN|>\", \"<|END_OF_TURN_TOKEN|>\"]:\n",
        "    tid = tokenizer.convert_tokens_to_ids(tok)\n",
        "    print(f\"Special token '{tok}' mapped to ID: {tid}\")\n",
        "\n",
        "def clean_vram():\n",
        "    \"\"\"Aggressively purges Python references, IPython output cache, and CUDA memory.\"\"\"\n",
        "    import gc, sys\n",
        "    sys.last_traceback = None\n",
        "    sys.last_type = None\n",
        "    sys.last_value = None\n",
        "    for _ in range(3):\n",
        "        gc.collect()\n",
        "    if torch.cuda.is_available():\n",
        "        for d in range(torch.cuda.device_count()):\n",
        "            with torch.cuda.device(d):\n",
        "                torch.cuda.empty_cache()\n",
        "                torch.cuda.ipc_collect()\n",
        "\n",
        "def print_vram_status(tag=\"\"):\n",
        "    if not torch.cuda.is_available():\n",
        "        return\n",
        "    print(f\"\\n--- VRAM Status [{tag}] ---\")\n",
        "    for d in range(torch.cuda.device_count()):\n",
        "        alloc = torch.cuda.memory_allocated(d) / (1024**3)\n",
        "        res = torch.cuda.memory_reserved(d) / (1024**3)\n",
        "        total = torch.cuda.get_device_properties(d).total_memory / (1024**3)\n",
        "        print(f\"  GPU {d}: Allocated={alloc:.2f} GB | Reserved={res:.2f} GB | Total={total:.2f} GB\")\n",
        "\n",
        "class SFTConversationDataset(Dataset):\n",
        "    def __init__(self, data, tokenizer, max_length=1024):\n",
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
        "        \n",
        "        # Guard: verify active targets\n",
        "        active = sum(1 for l in labels if l != -100)\n",
        "        if active == 0:\n",
        "            raise ValueError(f\"Sample {idx} has 0 supervised assistant tokens after max_length={self.max_length} truncation.\")\n",
        "            \n",
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
        "train_dataset = SFTConversationDataset(train_samples, tokenizer, max_length=1024)\n",
        "print(f\"Prepared dataset with {len(train_dataset)} examples.\")\n",
        "\n",
        "def evaluate_on_val(model, tok, run_name, device):\n",
        "    results = []\n",
        "    tool_calls_count = 0\n",
        "    successes = 0\n",
        "    print(f\"\\n--- Running Validation Rollout: {run_name} on {device} ---\")\n",
        "    \n",
        "    canonical_prompt = train_samples[0][0][\"value\"]\n",
        "    model.eval()\n",
        "    \n",
        "    for idx, t in enumerate(val_tasks):\n",
        "        q = t[\"user_query\"]\n",
        "        expected_ans = t[\"oracle_answer\"]\n",
        "        \n",
        "        prompt = f\"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{canonical_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{q}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\"\n",
        "        enc = tok(prompt, return_tensors=\"pt\").to(device)\n",
        "        \n",
        "        with torch.no_grad():\n",
        "            out = model.generate(\n",
        "                **enc,\n",
        "                max_new_tokens=128,\n",
        "                do_sample=False,\n",
        "                pad_token_id=tok.pad_token_id or tok.eos_token_id,\n",
        "                eos_token_id=tok.convert_tokens_to_ids(\"<|END_OF_TURN_TOKEN|>\"),\n",
        "            )\n",
        "        \n",
        "        gen_text = tok.decode(out[0][enc[\"input_ids\"].shape[1]:], skip_special_tokens=False)\n",
        "        has_action = \"<|START_ACTION|>\" in gen_text and \"<|END_ACTION|>\" in gen_text\n",
        "        if has_action:\n",
        "            tool_calls_count += 1\n",
        "            \n",
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
        "        del enc, out\n",
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
        "def train_model(model, run_name, device, max_steps=400, lr=1e-4, timeout_minutes=75):\n",
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
        "    print(f\"\\nStarting Training: {run_name} on {device} (max_steps={max_steps}, lr={lr}, timeout={timeout_minutes}m)...\")\n",
        "    while step < max_steps:\n",
        "        for batch in loader:\n",
        "            if (time.time() - start_time) > (timeout_minutes * 60):\n",
        "                raise TimeoutError(f\"Training exceeded {timeout_minutes}-minute wall-clock cap!\")\n",
        "                \n",
        "            batch = {k: v.to(device) for k, v in batch.items()}\n",
        "            outputs = model(**batch)\n",
        "            loss = outputs.loss / accum_steps\n",
        "            loss_val = outputs.loss.item()\n",
        "            \n",
        "            if not math.isfinite(loss_val):\n",
        "                raise ValueError(f\"CRITICAL: Non-finite loss encountered at step {step}: {loss_val}\")\n",
        "            if loss_val > 15.0:\n",
        "                raise ValueError(f\"CRITICAL: Loss explosion threshold exceeded at step {step}: {loss_val}\")\n",
        "                \n",
        "            loss.backward()\n",
        "            \n",
        "            # Immediately free GPU tensors to avoid memory fragmentation\n",
        "            del batch, outputs, loss\n",
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
        "    del loader, optimizer, scheduler\n",
        "    return model\n"
    ]
}
cells.append(cell_4)

# ==============================================================================
# Cell 5: Matched Untuned Base Model Validation Rollout
# ==============================================================================
cell_5 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 5. Matched Untuned Base Model Validation Rollout (val_base_aya_3b_matched)\n",
        "clean_vram()\n",
        "print_vram_status(\"Pre-Base Rollout\")\n",
        "\n",
        "bnb_config = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_use_double_quant=True,\n",
        "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
        ")\n",
        "\n",
        "print(\"Loading untuned CohereLabs/tiny-aya-global on GPU 0...\")\n",
        "base_model = AutoModelForCausalLM.from_pretrained(\n",
        "    MODEL_ID,\n",
        "    quantization_config=bnb_config,\n",
        "    device_map={\"\": 0},\n",
        "    token=hf_token,\n",
        ")\n",
        "base_model.eval()\n",
        "\n",
        "base_results, base_m1, base_m2 = evaluate_on_val(base_model, tokenizer, \"val_base_aya_3b_matched\", device=\"cuda:0\")\n",
        "\n",
        "# Full cleanup to free GPU 0 for Condition C2\n",
        "print(\"\\nCleaning up Base Model from GPU 0...\")\n",
        "del base_model\n",
        "clean_vram()\n",
        "print_vram_status(\"Post-Base Rollout\")\n"
    ]
}
cells.append(cell_5)

# ==============================================================================
# Cell 6: Condition C2 Training (Attention-Only LoRA)
# ==============================================================================
cell_6 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 6. Condition C2 Training (Attention-Only LoRA: q, k, v, o)\n",
        "clean_vram()\n",
        "print_vram_status(\"Pre-C2 Setup\")\n",
        "\n",
        "bnb_config_c2 = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_use_double_quant=True,\n",
        "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
        ")\n",
        "\n",
        "print(\"Loading base model on GPU 0 for Condition C2...\")\n",
        "base_model_c2 = AutoModelForCausalLM.from_pretrained(\n",
        "    MODEL_ID,\n",
        "    quantization_config=bnb_config_c2,\n",
        "    device_map={\"\": 0},\n",
        "    token=hf_token,\n",
        ")\n",
        "base_model_c2 = prepare_model_for_kbit_training(base_model_c2, gradient_checkpointing_kwargs={\"use_reentrant\": False})\n",
        "\n",
        "lora_config_c2 = LoraConfig(\n",
        "    r=16,\n",
        "    lora_alpha=32,\n",
        "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],\n",
        "    lora_dropout=0.05,\n",
        "    bias=\"none\",\n",
        "    task_type=\"CAUSAL_LM\",\n",
        ")\n",
        "model_c2 = get_peft_model(base_model_c2, lora_config_c2)\n",
        "model_c2.print_trainable_parameters()\n",
        "\n",
        "# Train Condition C2 (400 steps)\n",
        "model_c2 = train_model(model_c2, \"sft_aya_3b_v2_c2\", device=\"cuda:0\", max_steps=400, lr=1e-4, timeout_minutes=45)\n",
        "\n",
        "# Evaluate Condition C2\n",
        "model_c2.eval()\n",
        "c2_results, c2_m1, c2_m2 = evaluate_on_val(model_c2, tokenizer, \"sft_aya_3b_v2_c2\", device=\"cuda:0\")\n",
        "\n",
        "# Full cleanup to ensure no residual memory on GPU 0\n",
        "print(\"\\nCleaning up Condition C2 model...\")\n",
        "del model_c2, base_model_c2\n",
        "clean_vram()\n",
        "print_vram_status(\"Post-C2 Cleanup\")\n"
    ]
}
cells.append(cell_6)

# ==============================================================================
# Cell 7: Condition C3 Training (All-Linear LoRA)
# ==============================================================================
cell_7 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 7. Condition C3 Training (All-Linear LoRA: q, k, v, o, gate, up, down)\n",
        "clean_vram()\n",
        "print_vram_status(\"Pre-C3 Setup\")\n",
        "\n",
        "# Select target GPU: GPU 1 if dual-GPU available (Kaggle T4x2), else GPU 0\n",
        "c3_device_id = 1 if torch.cuda.device_count() > 1 else 0\n",
        "c3_device = f\"cuda:{c3_device_id}\"\n",
        "torch.cuda.set_device(c3_device_id)\n",
        "print(f\"Condition C3 will execute on: {c3_device}\")\n",
        "\n",
        "bnb_config_c3 = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_use_double_quant=True,\n",
        "    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 prevents SwiGLU activation overflow\n",
        ")\n",
        "\n",
        "print(f\"Loading base model on {c3_device} for Condition C3...\")\n",
        "base_model_c3 = AutoModelForCausalLM.from_pretrained(\n",
        "    MODEL_ID,\n",
        "    quantization_config=bnb_config_c3,\n",
        "    device_map={\"\": c3_device_id},\n",
        "    token=hf_token,\n",
        ")\n",
        "base_model_c3 = prepare_model_for_kbit_training(base_model_c3, gradient_checkpointing_kwargs={\"use_reentrant\": False})\n",
        "\n",
        "lora_config_c3 = LoraConfig(\n",
        "    r=16,\n",
        "    lora_alpha=32,\n",
        "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
        "    lora_dropout=0.05,\n",
        "    bias=\"none\",\n",
        "    task_type=\"CAUSAL_LM\",\n",
        ")\n",
        "model_c3 = get_peft_model(base_model_c3, lora_config_c3)\n",
        "model_c3.print_trainable_parameters()\n",
        "\n",
        "# Train Condition C3 with 75-minute timeout guard\n",
        "model_c3 = train_model(model_c3, \"sft_aya_3b_v2_c3\", device=c3_device, max_steps=400, lr=1e-4, timeout_minutes=75)\n",
        "\n",
        "# Evaluate Condition C3\n",
        "model_c3.eval()\n",
        "c3_results, c3_m1, c3_m2 = evaluate_on_val(model_c3, tokenizer, \"sft_aya_3b_v2_c3\", device=c3_device)\n",
        "\n",
        "# Clean up Condition C3\n",
        "print(\"\\nCleaning up Condition C3 model...\")\n",
        "del model_c3, base_model_c3\n",
        "clean_vram()\n",
        "print_vram_status(\"Post-C3 Cleanup\")\n"
    ]
}
cells.append(cell_7)

# ==============================================================================
# Cell 8: Comparative Results Table & Capability Floor Check
# ==============================================================================
cell_8 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 8. Comparative Results Table & Capability Floor Check\n",
        "from pathlib import Path\n",
        "import json\n",
        "\n",
        "runs_dir = Path(\"results/runs\")\n",
        "\n",
        "def load_metrics(run_name):\n",
        "    trace_path = runs_dir / run_name / \"val_rollout_traces.json\"\n",
        "    if not trace_path.exists():\n",
        "        return None, None\n",
        "    data = json.loads(trace_path.read_text(encoding=\"utf-8\"))\n",
        "    m1 = sum(1 for x in data if x.get(\"is_success\")) / len(data) if data else 0.0\n",
        "    m2 = sum(1 for x in data if x.get(\"has_tool_action\")) / len(data) if data else 0.0\n",
        "    return m1, m2\n",
        "\n",
        "base_m1, base_m2 = load_metrics(\"val_base_aya_3b_matched\")\n",
        "c2_m1, c2_m2 = load_metrics(\"sft_aya_3b_v2_c2\")\n",
        "c3_m1, c3_m2 = load_metrics(\"sft_aya_3b_v2_c3\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*75)\n",
        "print(\"             LEXSI TRACK 3 — M5 CAPABILITY REMEDIATION EVALUATION          \")\n",
        "print(\"=\"*75)\n",
        "print(f\"{'Condition / Model':<32} | {'M1 Success':<12} | {'M2 Tool Call':<12} | {'Floor Status'}\")\n",
        "print(\"-\" * 75)\n",
        "print(f\"{'Historical M4 (sft_aya_3b_001)':<32} | {'0.0%':<12} | {'0.0%':<12} | {'FAILED (Early Termination)'}\")\n",
        "\n",
        "if base_m1 is not None:\n",
        "    print(f\"{'Untuned Base (Matched Comparator)':<32} | {f'{base_m1*100:.1f}%':<12} | {f'{base_m2*100:.1f}%':<12} | {'Baseline Comparator'}\")\n",
        "else:\n",
        "    print(f\"{'Untuned Base (Matched Comparator)':<32} | {'N/A':<12} | {'N/A':<12} | {'Pending'}\")\n",
        "\n",
        "if c2_m1 is not None:\n",
        "    c2_status = 'PASSED FLOOR' if c2_m1 >= 0.40 and c2_m2 >= 0.50 else 'BELOW FLOOR'\n",
        "    print(f\"{'C2: SFT v2 (Attention-only)':<32} | {f'{c2_m1*100:.1f}%':<12} | {f'{c2_m2*100:.1f}%':<12} | {c2_status}\")\n",
        "else:\n",
        "    print(f\"{'C2: SFT v2 (Attention-only)':<32} | {'N/A':<12} | {'N/A':<12} | {'Pending'}\")\n",
        "\n",
        "if c3_m1 is not None:\n",
        "    c3_status = 'PASSED FLOOR' if c3_m1 >= 0.40 and c3_m2 >= 0.50 else 'BELOW FLOOR'\n",
        "    print(f\"{'C3: SFT v2 (All-Linear LoRA)':<32} | {f'{c3_m1*100:.1f}%':<12} | {f'{c3_m2*100:.1f}%':<12} | {c3_status}\")\n",
        "else:\n",
        "    print(f\"{'C3: SFT v2 (All-Linear LoRA)':<32} | {'N/A':<12} | {'N/A':<12} | {'Pending'}\")\n",
        "\n",
        "print(\"=\"*75)\n",
        "\n",
        "summary_data = {\n",
        "    \"val_base_aya_3b_matched\": {\"m1\": base_m1, \"m2\": base_m2},\n",
        "    \"sft_aya_3b_v2_c2\": {\"m1\": c2_m1, \"m2\": c2_m2},\n",
        "    \"sft_aya_3b_v2_c3\": {\"m1\": c3_m1, \"m2\": c3_m2},\n",
        "}\n",
        "with open(runs_dir / \"m5_comparison_summary.json\", \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(summary_data, f, indent=2)\n",
        "print(\"Summary metrics saved to results/runs/m5_comparison_summary.json\")\n"
    ]
}
cells.append(cell_8)

# ==============================================================================
# Cell 9: Package Artifacts Archive
# ==============================================================================
cell_9 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 9. Package Artifacts Archive\n",
        "import os, hashlib, zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "print(\"Packaging M5 evaluation artifacts...\")\n",
        "!zip -r m5_kaggle_artifacts.zip results/runs/ data/sft/ configs/manifests/\n",
        "\n",
        "if not os.path.exists(\"m5_kaggle_artifacts.zip\"):\n",
        "    print(\"Standard zip CLI unavailable, packaging with Python zipfile...\")\n",
        "    with zipfile.ZipFile(\"m5_kaggle_artifacts.zip\", \"w\", zipfile.ZIP_DEFLATED) as zf:\n",
        "        for target_dir in [Path(\"results/runs\"), Path(\"data/sft\"), Path(\"configs/manifests\")]:\n",
        "            if target_dir.exists():\n",
        "                for p in target_dir.rglob(\"*\"):\n",
        "                    if p.is_file():\n",
        "                        zf.write(p, p.relative_to(Path(\".\")))\n",
        "\n",
        "if os.path.exists(\"m5_kaggle_artifacts.zip\"):\n",
        "    sz_mb = os.path.getsize(\"m5_kaggle_artifacts.zip\") / (1024 * 1024)\n",
        "    sha256 = hashlib.sha256(open(\"m5_kaggle_artifacts.zip\", \"rb\").read()).hexdigest()\n",
        "    print(f\"\\nSUCCESS: m5_kaggle_artifacts.zip created ({sz_mb:.2f} MB)\")\n",
        "    print(f\"SHA-256: {sha256}\")\n",
        "    print(\"\\nNext step: Download m5_kaggle_artifacts.zip from Kaggle Output to sync with local repository.\")\n",
        "else:\n",
        "    print(\"WARNING: m5_kaggle_artifacts.zip was not generated!\")\n"
    ]
}
cells.append(cell_9)

# Assemble final notebook dictionary
nb_dict = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "accelerator": "GPU"
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": cells
}

out_path = ROOT_DIR / "notebooks" / "kaggle_m5_training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=2)

print(f"Successfully generated {out_path} with {len(cells)} cells.")
print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
