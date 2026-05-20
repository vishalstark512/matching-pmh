#!/usr/bin/env python3
"""Write handcrafted docs/tasks/*.md and notebooks/tasks/*.ipynb for all 13 paper tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pmh.paper_tasks import PAPER_TASK_IDS, get_paper_task, list_paper_tasks

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "tasks"
NB = REPO / "notebooks" / "tasks"

HANDCRAFTED = frozenset(PAPER_TASK_IDS)
SKIP_NOTEBOOK = frozenset({"t01-classical"})  # multi-subtask sklearn hub (hand-maintained body)
SKIP_DOC = frozenset({"t01-classical", "t02a-vit-isotropic", "t02b-chexpert-isotropic"})

# GitHub-flavored math in markdown (use in f-strings via _SIGMA_TASK_MD — avoid "\\text" → tab).
_SIGMA_TASK_MD = r"$\Sigma_{\text{task}}$"

# Every generated task notebook uses these section headers (tests enforce).
STANDARD_SECTIONS: tuple[str, ...] = (
    "## 1 — Install",
    "## 2 — Config & imports",
    "## 3 — Load demo data",
    "## 4 — Scope (applicability)",
    "## 5 — Estimate " + _SIGMA_TASK_MD + " + PMH train",
    "## 6 — Step 5 (deploy holdout)",
    "## 7 — Paper reproduction",
    "## 8 — Your pipeline",
)

_HF_LOAD_DEMO = """import json, tempfile
from pathlib import Path

import torch
from pmh.integrations.huggingface import load_style_pairs_jsonl

rows = [
    {
        "id": "ex1",
        "prompt": "Summarize the paper.",
        "content_fixed": "The method matches deployment nuisance covariance.",
        "style_variants": {
            "bulleted": "- Matches Sigma_task\\n- Adds PMH penalty",
            "verbose": "In this detailed response, we explain matching at length.",
        },
    },
]
path = Path(tempfile.mkstemp(suffix=".jsonl")[1])
with path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\\n")

pairs = load_style_pairs_jsonl(path)

class HashEncoder(torch.nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        b, t = input_ids.shape
        h = torch.zeros(b, t, self.dim)
        for i in range(b):
            h[i] = torch.nn.functional.one_hot(input_ids[i] % self.dim, self.dim).float().mean(0)
        return type("Out", (), {"hidden_states": (h,)})()

class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
        rows = [[hash(w) % 997 for w in t.split()[:max_length]] or [0] for t in texts]
        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        mask = torch.zeros(len(rows), max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

encoder = HashEncoder(64)
tokenizer = ToyTokenizer()
print(len(pairs), "style pairs loaded")"""

_HF_ESTIMATE = """from pmh.integrations.huggingface import estimate_style_sigma

rank = preset.default_rank if preset else 8
artifact = estimate_style_sigma(pairs, encoder, tokenizer, rank=rank, batch_size=4)
print("preflight", artifact.preflight, "trace", artifact.sigma.trace().item())"""


def _md(lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines.splitlines()]}


def _code(lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [l + "\n" for l in lines.splitlines()],
    }


def _nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


# Doc bodies keyed by task_id (FINAL.md summaries)
DOC_EXTRA: dict[str, dict] = {
    "t02a-vit-isotropic": {
        "final": "paper_code/T2/Task2A/FINAL.md",
        "headline": "Isotropic PMH on ViT-B/16: +4.29 pp mean ImageNet-C, TDI −58% at σ=0.10.",
        "metrics": "",
        "nuisance_key": "isotropic",
        "preset": "t2a_vit_isotropic",
    },
    "t02b-chexpert-isotropic": {
        "final": "paper_code/T2/Task2B/FINAL.md",
        "headline": "Chest X-ray E1: best saliency (0.723) and ~9× lower embedding drift vs B0.",
        "metrics": "",
        "nuisance_key": "isotropic",
        "preset": "t2b_chexpert_isotropic",
    },
    "t03a-pose-gradient": {
        "final": "paper_code/T3/Task3A/FINAL.md",
        "headline": "E1_aniso subspace PMH on COCO pose: **54.49%** clean PCK@0.05 (+22.4 pp vs baseline 32.07%).",
        "metrics": "| baseline | E1_aniso |\n|----------|----------|\n| 32.07% PCK | **54.49%** PCK |\n| — | 35.21% @ occ 0.40 |",
        "nuisance_key": "augmentation",
        "preset": None,
    },
    "t03b-depth-augmentation": {
        "final": "paper_code/T3/Task3B/FINAL.md",
        "headline": "E1_aniso beats E1 on hard photometric depth stress (combined_hard AbsRel **0.2152** vs 0.2191).",
        "metrics": "| baseline | E1 | E1_aniso |\n|----------|-----|----------|\n| 0.2033 AbsRel | 0.1951 | **0.2152** (photo hard) |",
        "nuisance_key": "augmentation",
        "preset": "t3b_depth_d3",
    },
    "t04a-vision-domain": {
        "final": "paper_code/T4/Task4A/FINAL.md",
        "headline": "E1_multiscale Gram PMH on DomainNet real→sketch: **42.15%** test acc (+3.31 pp vs B0 38.84%).",
        "metrics": "| B0 | E1 | E1_multiscale |\n|----|-----|---------------|\n| 38.84% | 39.34% | **42.15%** |",
        "nuisance_key": "domain_shift",
        "preset": "t4_domain_d4",
        "note": (
            "Notebook = single-hook D4 with **class-aligned** Gram when loaders are `(x,y)`. "
            "Paper multiscale DomainNet: `paper_code/T4/Task4A/`."
        ),
    },
    "t04b-multilayer-vision": {
        "final": "paper_code/T4/Task4B/FINAL.md",
        "headline": "E1_multiscale rare-5 Cityscapes mIoU **30.75%** (+11.1 pp vs B0 19.68%).",
        "metrics": "| B0 | E1 | E1_multiscale |\n|----|-----|---------------|\n| 19.68% mIoU | 19.99% | **30.75%** |",
        "nuisance_key": "domain_shift",
        "preset": "t4_domain_d4",
        "note": (
            "Notebook runs `PMHTrainer(train_mode='feature_diff')` + `estimate_multilayer` on demo "
            "loaders. Paper GTA5→Cityscapes: `paper_code/T4/Task4B/`."
        ),
    },
    "t05a-qm9-molecule": {
        "final": "paper_code/T5/Task5A/FINAL.md",
        "headline": "E1 matched position PMH: clean MAE **24.921** (−0.155 vs B0); σ=0.2 Å noise MAE **47.415** vs B0.",
        "metrics": "| B0 MAE | E1 MAE | @ σ=0.2 |\n|--------|--------|--------|\n| 25.076 | **24.921** | **47.415** |",
        "nuisance_key": "compositional",
        "preset": "t5_compositional_d5",
    },
    "t05b-code-tokens": {
        "final": "paper_code/T5/Task5B/FINAL.md",
        "headline": "E1 identifier PMH: rename_bacc_ratio **0.9383** vs B0 **0.8297**.",
        "metrics": "| B0 | E1 | E1S (wrong blocks) |\n|----|-----|------------------|\n| 0.8297 | **0.9383** | 0.7379 (fails) |",
        "nuisance_key": "compositional",
        "preset": "t5_compositional_d5",
    },
    "t06a-speech-whisper": {
        "final": "paper_code/T6/task6A/FINAL.md",
        "headline": "pmh_content_residual: Libri **other-WER 14.63%** (−8.6 pp vs baseline 23.26%); TDI **0.381**.",
        "metrics": "| baseline WER | pmh_content_residual |\n|--------------|----------------------|\n| 23.26% | **14.63%** |",
        "nuisance_key": "temporal",
        "preset": None,
        "note": "Paper uses **content-residual W** (`pmh.calibrate.content_residual_subspace`). Demo uses `temporal` on sequence embeddings.",
    },
    "t06b-temporal-har": {
        "final": "paper_code/T6/task6B/FINAL.md",
        "headline": "Matched PMH wins HAR stress 3.0: bal. acc **0.4099** vs baseline **0.2794** (3 seeds).",
        "metrics": "| baseline | PMH | wrong_W |\n|----------|-----|--------|\n| 0.2794 @ stress 3 | **0.4099** | fails geometry |",
        "nuisance_key": "temporal",
        "preset": "t6_temporal_d6",
    },
    "t07a-llm-style": {
        "final": "paper_code/T7/task7A/FINAL.md",
        "headline": r"Matched $\Sigma_{\text{style}}$ RM: sycophancy **38.5%→13.5%**, style gap **2.199→0.803**; margin_pmh DPO Style TDI **1.836**.",
        "metrics": "| matched MC1 | sycophancy | style gap |\n|-------------|------------|----------|\n| 0.548 | **13.5%** | **0.803** |",
        "nuisance_key": "style",
        "preset": "t7a_style_d7",
    },
    "t07b-adversarial-pgd": {
        "final": "paper_code/T7/task7B/FINAL.md",
        "headline": "pmh_aniso (PGD-W): TDI **0.878** (−19% vs baseline 1.090); clean **80.9%**.",
        "metrics": "| baseline TDI | pmh_aniso TDI | clean acc |\n|--------------|---------------|----------|\n| 1.090 | **0.878** | **80.9%** |",
        "nuisance_key": "style",
        "preset": "t7b_pgd_d7",
    },
}


def render_doc_page(task_id: str) -> str:
    t = get_paper_task(task_id)
    extra = DOC_EXTRA.get(task_id, {})
    nb = f"../../{t.notebook}"
    lines = [
        f"# {t.block} — {t.title}",
        "",
        f"**Source of truth:** `{extra.get('final', 'paper_code/')}`",
        "",
        f"**Lemma:** {t.lemma} · **Stack:** {t.stack}",
        f"**Nuisance key:** `{extra.get('nuisance_key', t.lemma.lower())}`",
        "",
        f"**Production change:** {t.what_changes}",
        "",
        f"**Notebook (Run All, built-in demo):** [{task_id}.ipynb]({nb})",
        "",
        "```bash",
        t.install,
        "# Open the notebook and Run All",
        "```",
        "",
        f"## What this task achieved (headline)",
        "",
        f"> {extra.get('headline', t.what_changes)}",
        "",
        extra.get("metrics", ""),
        "",
    ]
    if t.preset:
        lines += [
            f"**Paper preset:** `{t.preset}` · `from pmh.benchmark.presets import get_preset`",
            "",
        ]
    if extra.get("note"):
        lines += [f"**Note:** {extra['note']}", ""]
    if t.subtasks:
        lines += ["## Subtasks (paper_code)", ""]
        for s in t.subtasks:
            lines += [
                f'<a id="{s.subtask_id}"></a>',
                "",
                f"### {s.title}",
                "",
                s.blurb,
                "",
                "```bash",
                f"python {s.script}" if s.script.endswith(".py") else f"# see {s.script}",
                "```",
                "",
            ]
            if s.preset:
                lines += [f"Preset: `{s.preset}`", ""]
    lines += [
        "## Run with matching-pmh",
        "",
        f"```python",
        f"from pmh import PMHTrainer, evaluate_robust_fit",
        f"# nuisance=\"{extra.get('nuisance_key', 'domain_shift')}\"",
        "```",
        "",
        "## Do not use PMH when",
        "",
        t.not_for,
        "",
        "## Replace demo data with yours",
        "",
        _replace_blurb(t),
        "",
        "[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)",
        "",
        f'<a id="{task_id}"></a>',
        "",
    ]
    return "\n".join(lines)


def _replace_blurb(t) -> str:
    if t.stack == "hf":
        return "Style-pair JSONL (same content, two surfaces) → `estimate_style_sigma` / D7 trainer."
    if t.stack == "sklearn":
        return "Frozen `features.npy` + labels per site → `evaluate_baseline_vs_pmh`."
    return (
        "Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, "
        "and deploy holdout. Hook the backbone before your task head."
    )


@dataclass(frozen=True)
class NotebookSpec:
    task_id: str
    demo: str  # domain | aug | comp | iso | seq | hf
    demo_note: str = ""
    falsification: bool = True
    extra_train: str = ""  # code before trainer.fit (estimate, augmentations, …)
    extra_after_step5: list[tuple[str, str]] = field(default_factory=list)  # (md, code)
    step5_include_falsification: bool | None = None
    step5_extra_kw: str = ""
    train_nuisance: str | None = None  # override for runnable demo (e.g. seq → domain_shift)
    train_mode: str = "jacobian"  # jacobian | feature_diff (T4B multilayer)


def _install_cmd(stack: str) -> str:
    if stack == "hf":
        return '!pip install -q "matching-pmh[hf]"'
    if stack == "sklearn":
        return '!pip install -q "matching-pmh[sklearn,vision]"'
    return "!pip install -q matching-pmh torch"


def _banner_md(task_id: str) -> str:
    t = get_paper_task(task_id)
    extra = DOC_EXTRA.get(task_id, {})
    nuisance = extra.get("nuisance_key", t.lemma.lower())
    final = extra.get("final", "paper_code/")
    headline = extra.get("headline", t.what_changes)
    demo_note = ""
    spec = NOTEBOOK_SPECS.get(task_id)
    if spec and spec.demo_note:
        demo_note = f"\n\n**Demo note:** {spec.demo_note}"
    return f"""# {t.block} — {t.title}

**Lemma {t.lemma}** · `nuisance="{nuisance}"` · [Task doc](../../docs/tasks/{task_id}.md) · FINAL: `{final}`

> {headline}

| § | What you do |
|---|-------------|
| 1–4 | Install → load demo → `check_applicability` |
| 5–6 | Estimate {_SIGMA_TASK_MD} → PMH train → Step 5 on deploy holdout |
| 7–8 | Reproduce paper scripts → plug in your data |
{demo_note}"""


def _load_demo_code(spec: NotebookSpec) -> str:
    t = get_paper_task(spec.task_id)
    extra = DOC_EXTRA.get(spec.task_id, {})
    preset = extra.get("preset") or t.preset
    preset_line = f'preset = get_preset("{preset}")' if preset else "preset = None"
    n_quick = "N = 120 if QUICK else 400" if spec.demo == "seq" else "N = 200 if QUICK else 500"
    batch = "batch_size=16" if spec.demo == "seq" else "batch_size=32"

    if spec.demo == "multilayer_vision":
        loader = "pytorch_multilayer_vision_demo_loaders"
        unpack = """model = bundle.model
hook, head = bundle.encoder, bundle.head
train_loader, src_loader, tgt_loader, val_loader = (
    bundle.train_loader, bundle.source_batches, bundle.target_batches, bundle.val_loader,
)
print("RGB multilayer demo", bundle.n_classes, "classes")"""
        return f"""{preset_line}
{n_quick}
bundle = {loader}(n=N, {batch}, seed=SEED)
{unpack}"""
    if spec.demo == "domain":
        loader = "pytorch_demo_loaders"
        unpack = """model = bundle.model
hook, head = bundle.encoder, bundle.head
train_loader, src_loader, tgt_loader, val_loader = (
    bundle.train_loader, bundle.source_batches, bundle.target_batches, bundle.val_loader,
)"""
    elif spec.demo == "iso":
        loader = "pytorch_isotropic_demo_loaders"
        sigma = "0.10" if spec.task_id == "t02a-vit-isotropic" else "0.08"
        unpack = f"""model = bundle.model
hook, head = bundle.encoder, bundle.head
train_loader, src_loader, val_loader = bundle.train_loader, bundle.source_batches, bundle.val_loader
tgt_loader = val_loader  # noisy deploy holdout (sigma={sigma})"""
        loader_call = f"{loader}(n=N, {batch}, seed=SEED, eval_noise_sigma={sigma})"
        return f"""{preset_line}
bundle = {loader_call}
{unpack}
print("demo", bundle.n_classes, "classes")"""
    elif spec.demo == "aug":
        loader = "pytorch_demo_loaders"
        unpack = """model = bundle.model
hook, head = bundle.encoder, bundle.head
train_loader, src_loader, tgt_loader, val_loader = (
    bundle.train_loader, bundle.source_batches, bundle.target_batches, bundle.val_loader,
)

def aug_noise(x):
    return x + 0.12 * torch.randn_like(x)

def aug_scale(x):
    return x * (0.9 + 0.2 * torch.rand(x.size(0), 1, device=x.device, dtype=x.dtype))

augmentations = [aug_noise, aug_scale]"""
    elif spec.demo == "comp":
        loader = "pytorch_demo_loaders"
        unpack = """model = bundle.model
hook, head = bundle.encoder, bundle.head
train_loader, src_loader, tgt_loader, val_loader = (
    bundle.train_loader, bundle.source_batches, bundle.target_batches, bundle.val_loader,
)
nuisance_indices = tuple(preset.estimate_kwargs.get("nuisance_indices", (0, 1, 2)))"""
    elif spec.demo == "seq":
        loader = "pytorch_sequence_demo_loaders"
        unpack = """model = bundle.model
hook, head = bundle.encoder, bundle.head
train_loader, src_loader, val_loader = (
    bundle.train_loader, bundle.sequence_batches, bundle.val_loader,
)
tgt_loader = val_loader"""
    elif spec.demo == "hf":
        return f"""{preset_line}
{_HF_LOAD_DEMO}"""
    else:
        raise ValueError(spec.demo)

    if spec.demo != "iso":
        return f"""{preset_line}
{n_quick}
bundle = {loader}(n=N, {batch}, seed=SEED)
{unpack}"""

    return ""  # iso returns early above


def _scope_code(spec: NotebookSpec) -> str:
    t = get_paper_task(spec.task_id)
    extra = DOC_EXTRA.get(spec.task_id, {})
    nuisance = spec.train_nuisance or extra.get("nuisance_key", "domain_shift")
    if spec.demo == "iso":
        return f"""from pmh import check_applicability, suggest_nuisance

print(suggest_nuisance(has_source_labels=True, has_target_domain=False))
app = check_applicability(stack="pytorch", has_target_domain=False)
print(app.summary())"""
    if spec.demo == "hf":
        return """from pmh import check_applicability

app = check_applicability(stack="hf", has_style_pairs=True)
print(app.summary())"""
    return f"""from pmh import check_applicability, suggest_nuisance

print(suggest_nuisance(has_source_labels=True, has_target_domain=True))
app = check_applicability(stack="pytorch", has_target_domain=True)
print(app.summary())
print("suggested nuisance:", app.suggested_nuisance, "(expect {nuisance!r})")"""


def _train_multilayer_code(spec: NotebookSpec) -> str:
    extra = DOC_EXTRA.get(spec.task_id, {})
    preset = extra.get("preset") or get_paper_task(spec.task_id).preset
    pmh_cfg = "pmh_config=preset.pmh_config" if preset else "pmh_config=PMHConfig.balanced()"
    rank_kw = "rank=preset.default_rank, " if preset else "rank=16, "
    if spec.demo == "multilayer_vision":
        ff_block = """layer_names = ("conv1", "conv2")
forward_features = m.forward_features"""
        hook_head = "hook=bundle.encoder, head=m.head"
    else:
        ff_block = """def forward_features(x):
    z = m.enc[0](x)
    h = m.enc(x)
    return {"layer0": z, "layer2": h}

layer_names = ("layer0", "layer2")"""
        hook_head = "hook=m.enc, head=m.head"
    return f"""import copy
from pmh import PMHTrainer, PMHConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = copy.deepcopy(model).to(device)
{ff_block}
trainer = PMHTrainer(
    m, {hook_head}, nuisance="domain_shift", {rank_kw}{pmh_cfg},
    train_mode="feature_diff", forward_features=forward_features, layer_names=layer_names,
    head_layer=layer_names[-1], device=device,
)
sigmas = trainer.estimate_multilayer(src_loader, tgt_loader, max_batches=10 if QUICK else 30)
print("per-layer sigma shapes:", {{k: tuple(v.shape) for k, v in sigmas.items()}})
trainer.fit(
    train_loader, source_batches=src_loader, target_batches=tgt_loader,
    epochs=EPOCHS, max_steps_per_epoch=8 if QUICK else None,
)
print("feature_diff train done; preflight", trainer.artifact_.preflight)"""


def _train_code(spec: NotebookSpec) -> str:
    if spec.train_mode == "feature_diff":
        return _train_multilayer_code(spec)
    extra = DOC_EXTRA.get(spec.task_id, {})
    nuisance = spec.train_nuisance or extra.get("nuisance_key", "domain_shift")
    preset = extra.get("preset") or get_paper_task(spec.task_id).preset
    preset_ref = "preset" if preset else "PMHConfig.balanced()"
    pmh_cfg = "pmh_config=preset.pmh_config" if preset else "pmh_config=PMHConfig.balanced()"

    if spec.demo == "hf":
        return _HF_ESTIMATE

    rank_kw = ""
    if preset and nuisance != "isotropic":
        rank_kw = "rank=preset.default_rank, "
    elif nuisance != "isotropic":
        rank_kw = "rank=16, "

    extra_noise = ""
    if nuisance == "isotropic":
        extra_noise = "noise_level=preset.estimate_kwargs[\"noise_level\"], "

    nui_indices = ""
    if nuisance == "compositional":
        nui_indices = "nuisance_indices=nuisance_indices, "

    hook_head = "hook=hook, head=head"
    if spec.demo in ("domain", "multilayer_vision"):
        hook_head = "hook=m.enc, head=m.head" if spec.demo == "domain" else "hook=bundle.encoder, head=m.head"

    fit_args = "train_loader, source_batches=src_loader, target_batches=tgt_loader, epochs=EPOCHS"
    if spec.demo == "aug":
        fit_args = (
            "train_loader, source_batches=src_loader, epochs=EPOCHS  # estimate done above"
        )
    if spec.demo == "comp":
        fit_args = "train_loader, source_batches=src_loader, epochs=EPOCHS"
    if spec.demo == "iso":
        fit_args = "train_loader, source_batches=src_loader, epochs=EPOCHS"
    if spec.demo == "seq":
        fit_args = "train_loader, source_batches=src_loader, target_batches=src_loader, epochs=EPOCHS"

    pre = spec.extra_train.strip()
    if pre:
        pre = pre + "\n"

    return f"""import copy
from pmh import PMHTrainer, PMHConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = copy.deepcopy(model).to(device)
trainer = PMHTrainer(
    m, {hook_head}, nuisance="{nuisance}", {rank_kw}{nui_indices}{extra_noise}{pmh_cfg}, device=device,
)
{pre}trainer.fit({fit_args})
print("preflight", trainer.artifact_.preflight, "method", getattr(trainer.artifact_, "method", None))"""


def _step5_code(spec: NotebookSpec) -> str:
    if spec.demo == "hf":
        return """# Step 5 for HF: attach artifact to your causal LM, then run deployment eval.
# PMHTrainer.from_artifact(model, hook=last_hidden_state_layer, artifact=artifact, pmh_config=preset.pmh_config)
print("artifact ready — plug into your HF training loop (see task doc §8)")"""

    extra = DOC_EXTRA.get(spec.task_id, {})
    nuisance = spec.train_nuisance or extra.get("nuisance_key", "domain_shift")
    preset = extra.get("preset") or get_paper_task(spec.task_id).preset
    pmh_cfg = "pmh_config=preset.pmh_config" if preset else "pmh_config=PMHConfig.balanced()"
    rank_kw = "rank=preset.default_rank, " if preset and nuisance != "isotropic" else "rank=16, "
    fals = spec.step5_include_falsification
    if fals is None:
        fals = spec.falsification
    fals_s = "True" if fals else "False"

    hook_head = "hook=m.enc, head=m.head"
    if spec.demo == "multilayer_vision":
        hook_head = "hook=bundle.encoder, head=m.head"
    if spec.demo == "seq":
        hook_head = "hook=hook, head=head"

    extra_kw = spec.step5_extra_kw
    if nuisance == "isotropic":
        extra_kw = (extra_kw + "\n    noise_level=preset.estimate_kwargs[\"noise_level\"],").strip()
    if nuisance == "compositional":
        extra_kw = (extra_kw + "\n    nuisance_indices=nuisance_indices,").strip()

    kw_block = f"\n    {extra_kw}" if extra_kw else ""

    return f"""from pmh import evaluate_robust_fit

report = evaluate_robust_fit(
    m, train_loader, val_loader,
    source_batches=src_loader, target_batches=tgt_loader,
    {hook_head}, nuisance="{nuisance}", {rank_kw}
    {pmh_cfg}, epochs=max(2, EPOCHS - 2), include_falsification={fals_s}, seed=SEED,{kw_block}
)
print(report.summary())
if hasattr(report, "baseline_metric"):
    print("deploy holdout — baseline:", report.baseline_metric, "pmh:", report.pmh_metric)"""


def _paper_md(spec: NotebookSpec) -> str:
    t = get_paper_task(spec.task_id)
    lines = [f"Frozen results: `{DOC_EXTRA.get(spec.task_id, {}).get('final', 'paper_code/')}`", ""]
    for s in t.subtasks[:4]:
        lines.append(f"- **{s.title}:** `python {s.script}`")
    if len(t.subtasks) > 4:
        lines.append(f"- … plus {len(t.subtasks) - 4} more subtasks in [task doc](../../docs/tasks/{spec.task_id}.md)")
    return "\n".join(lines)


def build_standard_notebook(spec: NotebookSpec) -> list[dict]:
    t = get_paper_task(spec.task_id)
    pytorch_imports = """import os
import torch
from pmh.benchmark.presets import get_preset
from pmh.pytorch_eval import (
    pytorch_demo_loaders,
    pytorch_isotropic_demo_loaders,
    pytorch_multilayer_vision_demo_loaders,
    pytorch_sequence_demo_loaders,
)
from pmh import PMHConfig, PMHTrainer, evaluate_robust_fit, check_applicability, suggest_nuisance
from pmh.adoption import RECIPE_ONE_LINER, format_recipe_banner

QUICK = os.environ.get("PMH_QUICK", "").lower() in ("1", "true", "yes")
EPOCHS = 2 if QUICK else 6
SEED = 0
print(RECIPE_ONE_LINER)
"""
    if spec.demo == "hf":
        pytorch_imports = """import os
import torch
from pmh.benchmark.presets import get_preset
from pmh import check_applicability
from pmh.adoption import RECIPE_ONE_LINER

QUICK = os.environ.get("PMH_QUICK", "").lower() in ("1", "true", "yes")
SEED = 0
print(RECIPE_ONE_LINER)
"""

    cells = [
        _md(_banner_md(spec.task_id)),
        _md(STANDARD_SECTIONS[0]),
        _code(_install_cmd(t.stack)),
        _md(STANDARD_SECTIONS[1]),
        _code(pytorch_imports),
        _md(STANDARD_SECTIONS[2]),
        _code(_load_demo_code(spec)),
        _md(STANDARD_SECTIONS[3]),
        _code(_scope_code(spec)),
        _md(STANDARD_SECTIONS[4]),
        _code(_train_code(spec)),
        _md(STANDARD_SECTIONS[5]),
        _code(_step5_code(spec)),
    ]
    for md, code in spec.extra_after_step5:
        cells.append(_md(md))
        cells.append(_code(code))
    cells.append(_md(STANDARD_SECTIONS[6]))
    cells.append(_md(_paper_md(spec)))
    cells.append(_md(STANDARD_SECTIONS[7]))
    cells.append(_md(_replace_blurb(t)))
    return cells


NOTEBOOK_SPECS: dict[str, NotebookSpec] = {
    "t02a-vit-isotropic": NotebookSpec(
        "t02a-vit-isotropic",
        "iso",
        demo_note="Mini RGB CNN (32×32), not full ImageNet ViT.",
        falsification=False,
        extra_after_step5=[
            (
                "### Geometry probe (paper §5)",
                """from pmh.tdi import trajectory_tdi_encoder
tdi = trajectory_tdi_encoder(m, hook, src_loader, sigma=0.10, max_batches=8, device=device)
print("trajectory_tdi:", tdi.get("trajectory_tdi"))""",
            ),
        ],
    ),
    "t02b-chexpert-isotropic": NotebookSpec(
        "t02b-chexpert-isotropic",
        "iso",
        demo_note="Medical-style σ=0.08 eval noise; full Pneumonia run in paper_code.",
        falsification=False,
        extra_after_step5=[
            (
                "### Embedding drift proxy (paper §4.3)",
                """import numpy as np
from pmh.features import collect_features
from pmh.tdi import tdi_feature_isotropic
m.eval()
h_clean = collect_features(hook, src_loader, max_batches=10, device=device).cpu().numpy()
parts = [hook(xb.to(device)).detach().cpu().numpy() for xb, _ in val_loader]
h_noisy = np.concatenate(parts, axis=0)
print("tdi proxy clean", tdi_feature_isotropic(h_clean, sigma=0.08))
print("tdi proxy noisy", tdi_feature_isotropic(h_noisy, sigma=0.08))""",
            ),
        ],
    ),
    "t03a-pose-gradient": NotebookSpec(
        "t03a-pose-gradient",
        "aug",
        extra_train="trainer.estimate(source_batches=src_loader, augmentations=augmentations)\n",
    ),
    "t03b-depth-augmentation": NotebookSpec(
        "t03b-depth-augmentation",
        "aug",
        extra_train="trainer.estimate(source_batches=src_loader, augmentations=augmentations)\n",
    ),
    "t04a-vision-domain": NotebookSpec(
        "t04a-vision-domain",
        "domain",
        demo_note=(
            "Runnable demo = single-hook D4 with class-aligned Gram on synthetic tabular shift. "
            "Paper DomainNet multiscale — §7 scripts."
        ),
    ),
    "t04b-multilayer-vision": NotebookSpec(
        "t04b-multilayer-vision",
        "multilayer_vision",
        demo_note=(
            "Runnable **feature_diff** on a tiny RGB CNN (conv1 + conv2): class-aligned per-layer Gram. "
            "Paper GTA5→Cityscapes — §7 scripts."
        ),
        train_mode="feature_diff",
        extra_after_step5=[
            (
                "### Golden path (deploy QA)",
                """from pmh import try_pmh

report = try_pmh(
    m, train_loader, val_loader,
    source_batches=src_loader, target_batches=tgt_loader,
    hook=bundle.encoder, head=m.head, epochs=2 if QUICK else 4,
)
print(report.deploy_summary())""",
            ),
        ],
    ),
    "t05a-qm9-molecule": NotebookSpec("t05a-qm9-molecule", "comp"),
    "t05b-code-tokens": NotebookSpec("t05b-code-tokens", "comp"),
    "t06a-speech-whisper": NotebookSpec(
        "t06a-speech-whisper",
        "seq",
        demo_note="Library D6 content-residual estimate (`d6_source='content'`). Paper Whisper/Libri — §7.",
        train_nuisance="temporal",
        extra_train="trainer.estimate(sequences_batches=src_loader, d6_source=\"content\")\n",
    ),
    "t06b-temporal-har": NotebookSpec(
        "t06b-temporal-har",
        "seq",
        demo_note="Library D6 (`d6_source='content'` default). Paper HAR consecutive-diff — §7.",
        train_nuisance="temporal",
        extra_train="trainer.estimate(sequences_batches=src_loader, d6_source=\"content\")\n",
    ),
    "t07a-llm-style": NotebookSpec("t07a-llm-style", "hf"),
    "t07b-adversarial-pgd": NotebookSpec(
        "t07b-adversarial-pgd",
        "domain",
        extra_after_step5=[
            (
                "### PGD subspace estimate (library)",
                """from pmh.calibrate.pgd import estimate_pgd_subspace_from_model

pgd_art = estimate_pgd_subspace_from_model(
    m, hook=m.enc, head=m.head, source_batches=src_loader,
    rank=8, epsilon=0.15, steps=2, max_batches=6 if QUICK else 20, device=device,
)
print("PGD artifact", pgd_art.method, "preflight", pgd_art.preflight)
# Full margin PMH + DPO: paper_code/T7/task7B/""",
            ),
        ],
    ),
}


# (task_id, deploy change, examples, estimate plain, nuisance=)
_TASK_GUIDE_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "t01-classical",
        "Frozen embeddings shift between sites",
        "Office-31; two hospitals’ features; lab A→B tabular",
        "Source−target subspace on features",
        "`subspace`",
    ),
    (
        "t02a-vit-isotropic",
        "Generic input noise / corruption",
        "ImageNet-C; camera noise; blur/JPEG",
        "Isotropic noise level σ",
        "`isotropic`",
    ),
    (
        "t02b-chexpert-isotropic",
        "Scanner / hospital appearance on X-ray",
        "CheXpert site shift; DICOM pipeline change",
        "Isotropic σ (medical deploy stress)",
        "`isotropic`",
    ),
    (
        "t03a-pose-gradient",
        "Camera/lighting; same keypoints",
        "Studio→in-the-wild pose; broadcast→fan photos",
        "Augmentation feature deltas",
        "`augmentation`",
    ),
    (
        "t03b-depth-augmentation",
        "Photometric shift; depth meaning fixed",
        "Lighting on depth maps; synthetic→real RGB-D",
        "Augmentation deltas",
        "`augmentation`",
    ),
    (
        "t04a-vision-domain",
        "New camera, site, or visual domain",
        "Photo→sketch; warehouse A→B; day→night cls",
        "Train vs deploy feature Gram",
        "`domain_shift`",
    ),
    (
        "t04b-multilayer-vision",
        "Sim→real texture + layout (segmentation)",
        "GTA5→Cityscapes; synthetic IR→real seg",
        "Domain Gram (multilayer in paper)",
        "`domain_shift`",
    ),
    (
        "t05a-qm9-molecule",
        "Atom positions move; property label fixed",
        "QM9 conformers; docked poses",
        "Nuisance coordinates (positions)",
        "`compositional`",
    ),
    (
        "t05b-code-tokens",
        "Token groups change; task label fixed",
        "Renames; comment strip; obfuscation",
        "Nuisance token/block indices",
        "`compositional`",
    ),
    (
        "t06a-speech-whisper",
        "Mic, room, codec — same words",
        "Libri conditions; new microphone",
        "Temporal / content-residual (see doc)",
        "`temporal`",
    ),
    (
        "t06b-temporal-har",
        "Sensor drift over time",
        "HAR placement; IMU aging",
        "Temporal residual on sequences",
        "`temporal`",
    ),
    (
        "t07a-llm-style",
        "Tone/format; facts unchanged",
        "Bulleted vs prose; formal vs casual bot",
        "Style pairs (same content)",
        "`style`",
    ),
    (
        "t07b-adversarial-pgd",
        "Adversarial perturbations at deploy",
        "PGD robustness; spoof patches",
        "Subspace from attack deltas",
        "`style` (PGD path)",
    ),
)


def render_index() -> str:
    lines = [
        "# 13 paper tasks (T1 → T7)",
        "",
        "Tasks are listed **in paper order**. Your pipeline does not need to match a paper ID — pick the row whose **deploy change** sounds like yours, open the notebook, Run All on the demo, then edit §8 with your data.",
        "",
        "Full examples and estimation detail: **[README — Find your deployment story](../../README.md#find-your-deployment-story-t1-through-t7)**.",
        "",
        f"Matching principle ([main.pdf](../../main.pdf)): estimate {_SIGMA_TASK_MD} → matched PMH on `h` → Step 5 (matched vs wrong vs isotropic on deploy holdout).",
        "",
        "| # | Task | Page | Notebook |",
        "|---|------|------|----------|",
    ]
    for i, t in enumerate(list_paper_tasks(), 1):
        short = t.title.split("—")[0].strip() if "—" in t.title else t.title
        lines.append(
            f"| {i} | **{t.block}** {short} "
            f"| [{t.task_id}.md]({t.task_id}.md) "
            f"| [{t.task_id}.ipynb](../../{t.notebook}) |"
        )
    lines += [
        "",
        "## Which task fits your deploy change?",
        "",
        "| Task | What changes at deploy | Examples | What we estimate | `nuisance=` |",
        "|------|------------------------|----------|------------------|-------------|",
    ]
    for task_id, deploy, examples, estimate, nuisance in _TASK_GUIDE_ROWS:
        block = get_paper_task(task_id).block
        lines.append(
            f"| **{block}** | {deploy} | {examples} | {estimate} | {nuisance} | "
            f"[{task_id}.md]({task_id}.md) |"
        )
    lines += [
        "",
        "**T1** bundles seven classical subtasks in one notebook. **T2–T7** map to `paper_code/T2` … `T7`. Clone any row for a *similar* deploy change — not only the benchmark named in the paper.",
        "",
        "Regenerate: `python scripts/render_handcrafted_tasks.py`",
        "",
        "[Quickstart](../QUICKSTART.md) · [Will PMH help?](../WHEN_PMH_HELPS.md) · [API](../api/index.md)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    NB.mkdir(parents=True, exist_ok=True)
    (DOCS / "index.md").write_text(render_index(), encoding="utf-8")
    print("wrote docs/tasks/index.md")

    for task_id in PAPER_TASK_IDS:
        if task_id not in SKIP_DOC:
            (DOCS / f"{task_id}.md").write_text(render_doc_page(task_id), encoding="utf-8")
            print("doc", task_id)
        if task_id in SKIP_NOTEBOOK:
            continue
        nb_spec = NOTEBOOK_SPECS.get(task_id)
        if nb_spec is None:
            raise KeyError(f"no NotebookSpec for {task_id}")
        cells = build_standard_notebook(nb_spec)
        (NB / f"{task_id}.ipynb").write_text(
            json.dumps(_nb(cells), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("nb", task_id)


if __name__ == "__main__":
    main()
