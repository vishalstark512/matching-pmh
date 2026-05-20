"""Route a real ML task to one PMH integration path (no doc maze)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Stack = Literal["pytorch", "sklearn", "hf"]
Verdict = Literal["use_pmh", "maybe", "skip_pmh"]

# Stable ids for CLI / Python
TASK_IDS = (
    "pose_or_keypoints",
    "vision_classification",
    "vision_detection",
    "vision_segmentation",
    "nlp_text_classification",
    "llm_style_or_format",
    "tabular_same_schema",
    "speech_or_audio",
    "frozen_embeddings_sklearn",
    "augmentation_robustness",
    "temporal_drift",
    "pytorch_lightning",
    "compositional_coordinates",
    "generic_pytorch",
)


@dataclass(frozen=True)
class TaskRoute:
    """One integration answer for a developer situation."""

    task_id: str
    title: str
    verdict: Verdict
    verdict_summary: str
    what_changes: str  # plain-language nuisance (no D4 jargon first)
    sounds_like: tuple[str, ...]
    stack: Stack
    lemma: str
    nuisance: str
    golden_path: str  # G1, G1b, G2, ...
    hook_hint: str
    data_you_need: str
    not_for: str
    install: str
    example_script: str
    doc_one_pager: str
    walkthrough: tuple[str, ...]
    snippet: str


def _catalog() -> dict[str, TaskRoute]:
    g1 = "G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)"
    g1b = "G1b — Lightning (docs/GOLDEN_PATHS.md#g1b)"
    g2 = "G2 — sklearn (docs/GOLDEN_PATHS.md#g2)"
    g3 = "G3 — HF corpora (docs/GOLDEN_PATHS.md#g3)"

    _pose_what = (
        "Camera / studio / hospital **look** changes (lighting, viewpoint, sensor) "
        "but each keypoint index still means the same body joint."
    )
    _pose_walk = (
        "Confirm deploy uses the **same skeleton** (keypoint names and count) as training.",
        "Install: pip install matching-pmh torch · pmh-train doctor",
        "Build loaders: labeled train on site A; site B batches for estimate (labels optional).",
        "Pick hook on the **backbone** (before heatmap / coordinate head): suggest_hook(model).",
        "Estimate + train: robust_fit(..., source_batches=A, target_batches=B) — **keep your pose loss**.",
        "Read preflight (pass/marginal); evaluate keypoint metric on a **deploy holdout**.",
        "Before production: falsification controls (matched vs wrong-W) — walkthrough 08.",
    )
    common_pose = TaskRoute(
        task_id="pose_or_keypoints",
        title="Pose / keypoints — new camera or site",
        verdict="use_pmh",
        verdict_summary=(
            "Good fit when keypoint definitions are identical across cameras/sites "
            "and you can pass unlabeled frames from the deploy camera."
        ),
        what_changes=_pose_what,
        sounds_like=(
            "Fine-tuning pose on studio A, deploying on hospital camera B",
            "Same 17 COCO keypoints, different RGB / viewpoint",
            "Depth/RGB pose model, new scanner site",
        ),
        stack="pytorch",
        lemma="D4",
        nuisance="domain_shift",
        golden_path=g1,
        hook_hint="Backbone / encoder **before** the heatmap or regression head (not the final 2D output).",
        data_you_need=(
            "Labeled pose data from training site; unlabeled (or labeled) batches from deploy site "
            "in the same skeleton format."
        ),
        not_for="Different keypoint sets, new body parts at deploy, or no frames from deploy camera.",
        install="pip install matching-pmh torch",
        example_script="examples/00_first_run_domain_shift.py",
        doc_one_pager="docs/APPLICATIONS.md#pose_or_keypoints",
        walkthrough=_pose_walk,
        snippet=(
            "from pmh import check_applicability, robust_fit, suggest_hook\n"
            "print(check_applicability(stack='pytorch', n_source=N_TRAIN, n_target=N_DEPLOY).summary())\n"
            "hook = suggest_hook(model, alias='backbone').hook\n"
            "out = robust_fit(model, train_loader, source_batches=src, target_batches=deploy, hook=hook, epochs=E)"
        ),
    )

    return {
        "pose_or_keypoints": common_pose,
        "vision_classification": TaskRoute(
            task_id="vision_classification",
            title="Image classification — new camera or site",
            verdict="use_pmh",
            verdict_summary="Default path for train-on-A / deploy-on-B with the same class names.",
            what_changes="Image **appearance** shifts (camera, geography, device) while class names and meanings stay fixed.",
            sounds_like=(
                "ResNet/ViT trained on ImageNet site A, tested on phone photos",
                "Medical imaging: scanner A train, scanner B deploy",
                "Retail SKUs: warehouse photos vs store photos",
            ),
            stack="pytorch",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g1,
            hook_hint="Classifier input or TIMM `encoder_timm(backbone)` — see suggest_hook(model).",
            data_you_need="Source train/val; target domain loader (labels optional for D4).",
            not_for="New classes only at test time; different label semantics per site.",
            install="pip install matching-pmh torch",
            example_script="examples/00_first_run_domain_shift.py",
            doc_one_pager="docs/APPLICATIONS.md#vision_classification",
            walkthrough=(
                "Same class list on train and deploy (spellings and semantics).",
                "Collect unlabeled (or labeled) images from deploy domain B.",
                "hook = backbone / pooler before classifier (suggest_hook).",
                "robust_fit(model, train_loader, source_batches=A, target_batches=B, hook=hook).",
                "Compare deploy accuracy ERM vs PMH on a holdout.",
                "Run controls before claiming PMH helped (walkthrough 08).",
            ),
            snippet=common_pose.snippet,
        ),
        "vision_detection": TaskRoute(
            task_id="vision_detection",
            title="Object detection — same classes, new domain",
            verdict="maybe",
            verdict_summary=(
                "PMH applies to the **shared backbone**; you wire source/target image loaders. "
                "Box heads and matching are your framework — start with backbone-only shift."
            ),
            what_changes="Scene **style** and sensor change; bounding-box class IDs unchanged.",
            sounds_like=("YOLO trained on daytime dashcam, deploy on night camera", "Same COCO classes, new country"),
            stack="pytorch",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g1,
            hook_hint="FPN or backbone output used by the detection head (not per-anchor losses).",
            data_you_need="Batches of images from train and deploy domains (same class list).",
            not_for="Different category sets per region without relabeling.",
            install="pip install matching-pmh torch",
            example_script="examples/00_first_run_domain_shift.py",
            doc_one_pager="docs/APPLICATIONS.md#vision_detection",
            walkthrough=(
                "Confirm same box class IDs on train and deploy.",
                "Hook = FPN or backbone tensor (not per-anchor head).",
                "source_batches / target_batches = image tensors per site (box labels not needed for estimate).",
                "robust_fit on detector — keep localization + classification losses.",
                "Evaluate mAP on deploy holdout vs ERM.",
                "Add B images if preflight marginal.",
                "Falsification controls before production.",
            ),
            snippet=common_pose.snippet,
        ),
        "vision_segmentation": TaskRoute(
            task_id="vision_segmentation",
            title="Segmentation — same classes, new domain",
            verdict="maybe",
            verdict_summary="Same as detection: penalize backbone/encoder `h`, keep pixel loss.",
            what_changes="Pixel **texture and sensor** shift; per-pixel class IDs (road, person, …) unchanged.",
            sounds_like=("Segmentation trained on city A maps, deploy on city B weather",),
            stack="pytorch",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g1,
            hook_hint="Encoder output before decoder (U-Net bottleneck, DeepLab ASPP input).",
            data_you_need="Paired domain image loaders; same label map.",
            not_for="New stuff classes at deploy only.",
            install="pip install matching-pmh torch",
            example_script="examples/00_first_run_domain_shift.py",
            doc_one_pager="docs/APPLICATIONS.md#vision_segmentation",
            walkthrough=(
                "Confirm same per-pixel class map on A and B.",
                "Hook = encoder or U-Net bottleneck (before decoder).",
                "source_batches / target_batches = images per site (unlabeled B OK).",
                "robust_fit — keep pixel CE / Dice loss unchanged.",
                "Evaluate mIoU on deploy holdout vs ERM.",
                "Tune rank or add B images if preflight marginal.",
                "Controls before production.",
            ),
            snippet=common_pose.snippet,
        ),
        "nlp_text_classification": TaskRoute(
            task_id="nlp_text_classification",
            title="Text classification — new corpus or channel",
            verdict="use_pmh",
            verdict_summary="Encoder hook + source/target text batches; same label set.",
            what_changes="**Wording and channel** shift (support tickets vs chat) but intent/label set fixed.",
            sounds_like=("BERT trained on email, deploy on chat snippets", "Same 5 intents, new product UI copy"),
            stack="pytorch",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g1,
            hook_hint="Transformer pooler or last hidden state before classification head.",
            data_you_need="Labeled source; target corpus (unlabeled OK for D4).",
            not_for="Topic drift that changes label meaning; new intent labels at deploy.",
            install='pip install "matching-pmh[hf]"',
            example_script="examples/08_hf_style_d7.py",
            doc_one_pager="docs/APPLICATIONS.md#nlp_text_classification",
            walkthrough=(
                "Freeze intent/label definitions across corpora A and B.",
                "pip install matching-pmh[hf]; build text loaders for A and B.",
                "Hook = pooler or [CLS] before classifier.",
                "robust_fit(..., source_batches=A, target_batches=B).",
                "Evaluate accuracy on labeled deploy holdout.",
                "Stop if new intents appear at deploy (label shift).",
                "Controls before rollout.",
            ),
            snippet=common_pose.snippet,
        ),
        "llm_style_or_format": TaskRoute(
            task_id="llm_style_or_format",
            title="LLM — format / tone / template shift (same facts)",
            verdict="use_pmh",
            verdict_summary="Use D7 style pairs when content is fixed but surface form changes.",
            what_changes="**Surface form** (markdown, bullets, JSON vs prose) — not the underlying facts or instructions.",
            sounds_like=(
                "Train on formal reports, deploy on chat-style answers",
                "Same QA pairs, customer changes prompt template",
                "DPO data in one format, production system wraps differently",
            ),
            stack="hf",
            lemma="D7",
            nuisance="style",
            golden_path=g3,
            hook_hint="Last hidden states on paired prompts (style JSONL).",
            data_you_need="Style pair JSONL or two corpora with matched content.",
            not_for="Factual drift, new knowledge at deploy, safety policy changes only.",
            install='pip install "matching-pmh[hf]"',
            example_script="examples/08_hf_style_d7.py",
            doc_one_pager="docs/APPLICATIONS.md#llm_style_or_format",
            walkthrough=(
                "Build style-pair JSONL: same content, two surfaces (formal vs chat, etc.).",
                "pip install matching-pmh[hf]; pmh-train doctor --stack hf",
                "PMHTrainer(..., nuisance='style'); estimate(style_jsonl=...).",
                "fit(SFT/DPO loader) — keep task/preference loss.",
                "Evaluate on deploy-formatted holdout (same facts).",
                "Use G3b if you must keep transformers.Trainer.",
                "Not for factual drift — only surface form.",
            ),
            snippet=(
                "from pmh import PMHTrainer, PMHConfig\n"
                "trainer = PMHTrainer(model, hook=hook, nuisance='style', pmh_config=PMHConfig.balanced())\n"
                "trainer.estimate(style_jsonl='pairs.jsonl', model_id=MODEL_ID)\n"
                "trainer.fit(train_loader, epochs=E)"
            ),
        ),
        "tabular_same_schema": TaskRoute(
            task_id="tabular_same_schema",
            title="Tabular / clinical — new hospital or cohort",
            verdict="use_pmh",
            verdict_summary="Often G2: frozen features per row, PMHMatcher then sklearn classifier.",
            what_changes="**Cohort / hospital distribution** in the same feature columns; disease definition unchanged.",
            sounds_like=(
                "EHR model trained hospital A, deploy hospital B",
                "Same lab columns, new country prevalence but same ICD codes",
            ),
            stack="sklearn",
            lemma="D1",
            nuisance="subspace",
            golden_path=g2,
            hook_hint="N/A — operate on feature matrix rows.",
            data_you_need="Feature matrix + labels on source; features from target cohort.",
            not_for="Different schemas, new columns only at deploy, label definition change.",
            install='pip install "matching-pmh[sklearn]"',
            example_script="examples/06_office31_sklearn.py",
            doc_one_pager="docs/APPLICATIONS.md#tabular_same_schema",
            walkthrough=(
                "Verify same column schema and label definitions on A and B.",
                "Build x_source, y_source (A) and x_target (B); labels on B help for D1.",
                "pip install matching-pmh[sklearn]",
                "PMHMatcher(nuisance='subspace') in Pipeline with classifier.",
                "evaluate_baseline_vs_pmh on B holdout (compare CORAL optional).",
                "Run falsification: matched vs wrong-W.",
                "Need neural fine-tune? → G1 instead of G2.",
            ),
            snippet=(
                "from pmh import evaluate_baseline_vs_pmh\n"
                "report = evaluate_baseline_vs_pmh(x_src, y_src, x_tgt, y_tgt, compare_to=('coral',))\n"
                "print(report.summary())"
            ),
        ),
        "speech_or_audio": TaskRoute(
            task_id="speech_or_audio",
            title="Speech / audio — new mic, room, or channel",
            verdict="use_pmh",
            verdict_summary="Encoder hook on spectrogram or wav2vec trunk; D4 domain shift.",
            what_changes="**Acoustic channel** (mic, room, codec) — same words / phoneme labels.",
            sounds_like=("ASR trained on studio mic, deploy on phone", "Bioacoustic sensor swap"),
            stack="pytorch",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g1,
            hook_hint="Acoustic encoder before CTC/classification head.",
            data_you_need="Source transcripts/labels; target-domain audio batches.",
            not_for="New vocabulary or language at deploy without relabeling.",
            install="pip install matching-pmh torch",
            example_script="examples/00_first_run_domain_shift.py",
            doc_one_pager="docs/APPLICATIONS.md#speech_or_audio",
            walkthrough=(
                "Same vocabulary / class labels on A and B.",
                "Hook = acoustic encoder before CTC or classifier.",
                "source_batches / target_batches = audio chunks per site.",
                "robust_fit — keep CTC/CE loss.",
                "Evaluate WER or accuracy on deploy holdout.",
                "More B audio if preflight marginal.",
                "Controls before shipping.",
            ),
            snippet=common_pose.snippet,
        ),
        "frozen_embeddings_sklearn": TaskRoute(
            task_id="frozen_embeddings_sklearn",
            title="Frozen embeddings (.npy) — adapt without training CNN",
            verdict="use_pmh",
            verdict_summary="Fastest path — no PyTorch training loop required.",
            what_changes="**Feature distribution** between sites; you already extracted h and won't fine-tune the encoder.",
            sounds_like=("ResNet embeddings .npy from two cameras", "Already ran inference on both sites"),
            stack="sklearn",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g2,
            hook_hint="N/A",
            data_you_need="source_features.npy, target_features.npy (+ labels on source).",
            not_for="Need to adapt the neural encoder itself (use G1 instead).",
            install='pip install "matching-pmh[sklearn]"',
            example_script="examples/06_office31_sklearn.py",
            doc_one_pager="docs/APPLICATIONS.md#frozen_embeddings_sklearn",
            walkthrough=(
                "One folder per site: features.npy (+ optional labels.npy) — DATA_LAYOUT.md.",
                "pmh-train estimate --source-dir site_a --target-dir site_b",
                "PMHMatcher(nuisance='domain_shift').fit(x_a, x_b) in Pipeline.",
                "Train classifier; test on target holdout.",
                "evaluate_baseline_vs_pmh for ERM vs PMH report.",
                "If only frozen features, gains may be small — see WHEN_PMH_HELPS.",
                "Need encoder training? → G1.",
            ),
            snippet=(
                "from pmh import PMHMatcher\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "pipe = Pipeline([('pmh', PMHMatcher().fit(x_s, x_t)), ('clf', LogisticRegression())])\n"
                "pipe.fit(x_s, y_s)"
            ),
        ),
        "augmentation_robustness": TaskRoute(
            task_id="augmentation_robustness",
            title="Known augmentations — robust to blur, color, crop, …",
            verdict="use_pmh",
            verdict_summary=(
                "You can list finite transforms and run them on training data; "
                "PMH estimates sensitivity along those modes (D3)."
            ),
            what_changes="**Named transforms** you apply in training (blur, JPEG, rotation policy) — not an unknown new camera.",
            sounds_like=(
                "Want robustness to blur + color jitter + noise we already use",
                "Photometric policies from T2-style aug stacks",
                "Finite list of torchvision transforms",
            ),
            stack="pytorch",
            lemma="D3",
            nuisance="augmentation",
            golden_path=g1,
            hook_hint="Encoder on clean input; collect `encode(x_aug) - encode(x)` per mode.",
            data_you_need="Labeled train set + code to apply each aug mode; stack aug deltas [M, d].",
            not_for="Unknown deploy camera with no relation to your aug list; use D4 domain_shift instead.",
            install="pip install matching-pmh torch",
            example_script="examples/18_augmentation_d3.py",
            doc_one_pager="docs/APPLICATIONS.md#augmentation_robustness",
            walkthrough=(
                "List aug modes explicitly (e.g. blur, brightness, noise).",
                "For each mode: delta_h = mean(encoder(x_aug) - encoder(x)) on a reference batch.",
                "estimate_from_config(SigmaTaskConfig.for_augmentation(), aug_deltas=stack).",
                "PMHLoss(artifact, pmh_config) added to your task loss in the training loop.",
                "Compare val metric with vs without PMH on **held-out clean** data.",
                "Different deploy camera not in aug list → use D4 with site B batches instead.",
                "Example: examples/18_augmentation_d3.py",
            ),
            snippet=(
                "from pmh import SigmaTaskConfig, estimate_from_config, PMHConfig, PMHLoss\n"
                "# aug_stack: [M, d] mean representation deltas per aug mode\n"
                "art = estimate_from_config(SigmaTaskConfig.for_augmentation(), aug_deltas=aug_stack)\n"
                "pmh_loss = PMHLoss(art, PMHConfig.balanced())\n"
                "# total_loss = task_loss + pmh_loss(h, ...) in training_step"
            ),
        ),
        "temporal_drift": TaskRoute(
            task_id="temporal_drift",
            title="Temporal drift — same patient / session label over time",
            verdict="use_pmh",
            verdict_summary="Sequences [N,T,d] with label fixed over time; D6 estimates drift directions.",
            what_changes="**Measurement drift over time** (sensor aging, progression) while entity-level label is fixed.",
            sounds_like=(
                "ICU vitals trajectory, same patient label",
                "Wearable window sequences, chronic condition class",
                "Layer-wise activations over training steps (T6B-style)",
            ),
            stack="pytorch",
            lemma="D6",
            nuisance="temporal",
            golden_path=g1,
            hook_hint="Sequence encoder output; use collect_sequence_features or [N,T,d] tensors.",
            data_you_need="Batches shaped [N, T, d] with fixed label per sequence; T≥2.",
            not_for="Independent snapshots with no time axis; use D4.",
            install="pip install matching-pmh torch",
            example_script="examples/00_first_run_domain_shift.py",
            doc_one_pager="docs/APPLICATIONS.md#temporal_drift",
            walkthrough=(
                "Organize data as sequences (same label across time index).",
                "collect_sequence_features(model, seq_loader, hook) or pass [N,T,d].",
                "estimate_from_config(SigmaTaskConfig.for_temporal(rank=R), sequences).",
                "Add PMH penalty in training; keep sequence classification loss.",
                "Evaluate on future-time holdout windows.",
                "Walkthrough: docs/walkthroughs/11-temporal-d6.md for API detail.",
                "Controls before claiming drift-specific gain.",
            ),
            snippet=(
                "from pmh import SigmaTaskConfig, estimate_from_config, collect_sequence_features\n"
                "# feats: [N, T, d] from your sequence loader\n"
                "art = estimate_from_config(SigmaTaskConfig.for_temporal(rank=16), feats)"
            ),
        ),
        "pytorch_lightning": TaskRoute(
            task_id="pytorch_lightning",
            title="PyTorch Lightning — keep your LightningModule",
            verdict="use_pmh",
            verdict_summary="Same nuisance as G1; wire Phase A estimate + PMHLoss in training_step.",
            what_changes="Same as your underlying task (usually **site / camera** D4) — Lightning is the training shell.",
            sounds_like=(
                "Already have LightningModule + Trainer",
                "Domain A/B loaders but want pl.Trainer callbacks",
            ),
            stack="pytorch",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g1b,
            hook_hint="Same as G1 backbone hook; use PMHLightningCallback for schedule.",
            data_you_need="source_batches + target_batches for Phase A; task loss in training_step.",
            not_for="Plain script with no Lightning — use G1 robust_fit instead.",
            install='pip install "matching-pmh[lightning]"',
            example_script="examples/09_lightning_module.py",
            doc_one_pager="docs/APPLICATIONS.md#pytorch_lightning",
            walkthrough=(
                "Pick underlying application above (e.g. vision_classification) for nuisance story.",
                "pip install matching-pmh[lightning]",
                "Phase A: h_src, h_tgt from backbone on A/B batches → estimate_from_config.",
                "Build PMHLoss(artifact, PMHConfig.balanced()).",
                "training_step: total = task_loss + add_pmh_to_loss(...).",
                "PMHLightningCallback.from_artifact for cap/warmup.",
                "Template: templates/matching-pmh-starter/lightning_g1b_minimal.py",
            ),
            snippet=(
                'pip install "matching-pmh[lightning]"\n'
                "from pmh.integrations.lightning import PMHLightningCallback, add_pmh_to_loss\n"
                "# see docs/GOLDEN_PATHS.md#g1b"
            ),
        ),
        "compositional_coordinates": TaskRoute(
            task_id="compositional_coordinates",
            title="Compositional features — nuisance in part of h",
            verdict="use_pmh",
            verdict_summary=(
                "You know which coordinates of h are nuisance (joints, tokens, atom blocks); "
                "PMH uses D5 with nuisance_indices."
            ),
            what_changes="**Only some dimensions** of the representation move with deploy shift (not the whole vector).",
            sounds_like=(
                "QM9: molecular coords vs invariant atom features",
                "Code tokens: style in certain positions only",
                "Pose: subset of joints carry site-specific bias",
            ),
            stack="pytorch",
            lemma="D5",
            nuisance="compositional",
            golden_path=g1,
            hook_hint="Full h vector with `nuisance_indices` listing coordinate ranges.",
            data_you_need="Feature matrix [N,d] + list of nuisance column indices.",
            not_for="Global camera shift with no index structure — use D4.",
            install="pip install matching-pmh torch",
            example_script="examples/16_qm9_molecule_d5.py",
            doc_one_pager="docs/APPLICATIONS.md#compositional_coordinates",
            walkthrough=(
                "Identify nuisance coordinates (e.g. positions 0:3, token blocks 128:256).",
                "Pass nuisance_indices to SigmaTaskConfig.for_compositional(...) or PMHMatcher.",
                "estimate_from_config / fit on features with labels on source.",
                "Keep task loss on full h; PMH targets indexed subspace only.",
                "Walkthroughs: 05 compositional, 14 QM9, 15 CodeBERT tokens.",
                "Controls before claiming compositional structure helped.",
                "See estimators/d5.md for lemma detail.",
            ),
            snippet=(
                "from pmh import SigmaTaskConfig, estimate_from_config\n"
                "cfg = SigmaTaskConfig.for_compositional(nuisance_indices=[0, 1, 2, 8, 9])\n"
                "art = estimate_from_config(cfg, feature_matrix)"
            ),
        ),
        "generic_pytorch": TaskRoute(
            task_id="generic_pytorch",
            title="Other PyTorch task (regression, multi-head, custom)",
            verdict="maybe",
            verdict_summary=(
                "If labels mean the same on A and B and you have a representation `h`, use G1. "
                "Otherwise PMH is not automatic — run the gate below."
            ),
            what_changes="Whatever **environmental factor** changes between train and deploy without changing your target definition.",
            sounds_like=("Custom regression head, new factory sensors", "Multi-task net, one deploy site shift"),
            stack="pytorch",
            lemma="D4",
            nuisance="domain_shift",
            golden_path=g1,
            hook_hint="Last shared representation before task-specific heads.",
            data_you_need="source_batches + target_batches from deploy environment.",
            not_for="Pure i.i.d. training with no deploy domain.",
            install="pip install matching-pmh torch",
            example_script="examples/00_first_run_domain_shift.py",
            doc_one_pager="docs/APPLICATIONS.md#generic_pytorch",
            walkthrough=(
                "One sentence: what changes at deploy without changing the target?",
                "check_applicability(n_source, n_target).",
                "Hook = last shared representation before task heads.",
                "robust_fit(..., source_batches=A, target_batches=B, hook='auto').",
                "Deploy metric vs ERM.",
                "Controls.",
                "See docs/APPLICATIONS.md#generic_pytorch for examples.",
            ),
            snippet=common_pose.snippet,
        ),
    }


def list_tasks() -> list[TaskRoute]:
    """All built-in task profiles (stable order)."""
    cat = _catalog()
    return [cat[tid] for tid in TASK_IDS if tid in cat]


def get_task(task_id: str) -> TaskRoute:
    """Lookup by id; raises KeyError with hints."""
    key = task_id.strip().lower().replace("-", "_")
    cat = _catalog()
    if key not in cat:
        hints = ", ".join(TASK_IDS)
        raise KeyError(f"Unknown task {task_id!r}. Choose one of: {hints}")
    return cat[key]


def explain_task(task_id: str) -> str:
    """Human-readable plan for one task (CLI and notebooks)."""
    from pmh.adoption import RECIPE_ONE_LINER

    r = get_task(task_id)
    lines = [
        RECIPE_ONE_LINER,
        "",
        f"Application: {r.title}",
        f"Fit: {r.verdict.upper()} — {r.verdict_summary}",
        "",
        "WHAT CHANGES (nuisance PMH adapts to):",
        f"  {r.what_changes}",
        "",
        "SOUNDS LIKE:",
        *[f"  • {s}" for s in r.sounds_like],
        "",
        f"Library mapping: subtype {r.lemma} · nuisance={r.nuisance!r} · {r.golden_path}",
        f"Hook: {r.hook_hint}",
        f"You need: {r.data_you_need}",
        f"Not for: {r.not_for}",
        "",
        "WALKTHROUGH:",
        *[f"  {i}. {s}" for i, s in enumerate(r.walkthrough, 1)],
        "",
        f"Install: {r.install}",
        f"Example script: {r.example_script}",
        f"Docs: {r.doc_one_pager}",
        "",
        "Snippet:",
        r.snippet,
    ]
    return "\n".join(lines)


def format_task_menu(*, short: bool = True) -> str:
    """Menu for wizard (short) or route --list (full)."""
    if not short:
        from pmh.applications import format_application_finder, format_shift_types

        return "\n".join([format_shift_types(), format_application_finder()])

    lines = [
        "Which application is closest? (full table: docs/APPLICATIONS.md)",
        "",
    ]
    for i, r in enumerate(list_tasks(), 1):
        tag = {"use_pmh": "YES", "maybe": "TRY", "skip_pmh": "NO"}[r.verdict]
        lines.append(f"  [{i}] {tag}  {r.title}")
        lines.append(f"       Changes: {r.what_changes[:72]}{'…' if len(r.what_changes) > 72 else ''}")
    lines.append("")
    lines.append("YES = usual fit · TRY = backbone/features first · pmh-train route --task <id>")
    return "\n".join(lines)


def route_from_wizard_choice(choice: str) -> str | None:
    """Map wizard menu key '1'..'N' to task_id."""
    keys = list(TASK_IDS)
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    if choice in keys:
        return choice
    return None


def search_applications(query: str) -> list[TaskRoute]:
    """Keyword search over titles, nuisances, and 'sounds like' phrases."""
    q = query.strip().lower()
    if not q:
        return []
    hits: list[TaskRoute] = []
    for r in list_tasks():
        blob = " ".join(
            [r.task_id, r.title, r.what_changes, r.verdict_summary, r.nuisance, r.lemma, *r.sounds_like]
        ).lower()
        if q in blob:
            hits.append(r)
    return hits


def format_search_results(query: str) -> str:
    """CLI-friendly search output."""
    hits = search_applications(query)
    if not hits:
        return f"No application matched {query!r}. Try: pose, camera, llm, hospital, augmentation, temporal"
    lines = [f"Matches for {query!r}:", ""]
    for r in hits:
        lines.append(f"  - {r.title}  ->  pmh-train route --task {r.task_id}")
        tail = "..." if len(r.what_changes) > 80 else ""
        lines.append(f"    {r.what_changes[:80]}{tail}")
    lines.append("")
    lines.append("Full walkthrough: pmh-train route --task <id>")
    return "\n".join(lines)
