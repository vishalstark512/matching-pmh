"""Canonical 13 paper tasks (T1–T7 blocks) for docs and notebooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperSubtask:
    """T1 experiment line inside FINAL.md (one page lists all; no extra doc files)."""

    subtask_id: str
    title: str
    script: str
    preset: str | None = None
    blurb: str = ""


@dataclass(frozen=True)
class PaperTask:
    """One paper block = one docs page + one notebook."""

    task_id: str
    block: str
    title: str
    lemma: str
    what_changes: str
    stack: str  # sklearn | pytorch | hf
    preset: str | None
    install: str
    not_for: str
    page: str
    notebook: str
    demo_script: str | None = None  # optional CLI helper (T1 batch jobs)

    @property
    def subtasks(self) -> tuple[PaperSubtask, ...]:
        return _SUBTASKS.get(self.task_id, ())


PAPER_TASK_IDS: tuple[str, ...] = (
    "t01-classical",
    "t02a-vit-isotropic",
    "t02b-chexpert-isotropic",
    "t03a-pose-gradient",
    "t03b-depth-augmentation",
    "t04a-vision-domain",
    "t04b-multilayer-vision",
    "t05a-qm9-molecule",
    "t05b-code-tokens",
    "t06a-speech-whisper",
    "t06b-temporal-har",
    "t07a-llm-style",
    "t07b-adversarial-pgd",
)

# T1 lines from paper_code/T1/classical_pmh/FINAL.md (anchors for CLI/tests only)
_SUBTASKS: dict[str, tuple[PaperSubtask, ...]] = {
    "t01-classical": (
        PaperSubtask(
            "t1-ridge-theorem",
            "Ridge Theorem 1 (closed-form invariance)",
            "paper_code/T1/classical_pmh/ridge_theory.py",
            None,
            "Synthetic Gaussian; matched MSE flat in σ_test.",
        ),
        PaperSubtask(
            "t1-oracle-mnist-fashion",
            "Oracle-W: MNIST / Fashion-MNIST × SVM, k-NN, logistic",
            "paper_code/T1/classical_pmh/run_benchmarks.py",
            None,
            "Strict B0 < E1_matched; E1_wrong ≈ B0.",
        ),
        PaperSubtask(
            "t1-dct-drift",
            "Data-driven W: DCT drift + soft k-NN",
            "paper_code/T1/classical_pmh/mnist_drift.py",
            None,
            "Cross-domain SVD; Fashion-MNIST flagship (+15 pp SVM).",
        ),
        PaperSubtask(
            "t1-baselines-coral",
            "PMH vs CORAL / LMNN / IRM",
            "paper_code/T1/classical_pmh/baselines.py",
            None,
            "Head-to-head on oracle-W and drift.",
        ),
        PaperSubtask(
            "t1-svhn",
            "SVHN real digits (SVM)",
            "paper_code/T1/classical_pmh/svhn_subspace.py",
            None,
            "E1_matched +20 pp vs B0.",
        ),
        PaperSubtask(
            "t1-ridge-tabular",
            "California Housing + ridge",
            "paper_code/T1/classical_pmh/ridge_tabular.py",
            None,
            "Real tabular; theorem behaviour on UCI.",
        ),
        PaperSubtask(
            "t1-office31",
            "Office-31 Amazon→DSLR (ResNet-18, SVM/kNN/logistic)",
            "scripts/demos/office31_sklearn.py",
            "t1_office31_sklearn",
            "Real DA; CORAL > PMH; PMH > B0 on SVM.",
        ),
    ),
    "t02a-vit-isotropic": (
        PaperSubtask(
            "t2a-vit-imagenet",
            "ImageNet ViT-B/16 + isotropic PMH (Type 2A)",
            "paper_code/T2/Task2A/train.py",
            "t2a_vit_isotropic",
            "PMH ≈ ERM clean; +4.29 pp mean ImageNet-C; TDI −58% at σ=0.10.",
        ),
        PaperSubtask(
            "t2a-geometry-tdi",
            "TDI / Jacobian probes (label-free)",
            "paper_code/T2/Task2A/recompute_task1a_tdi.py",
            None,
            "Layer-averaged CLS displacement under input Gaussian.",
        ),
        PaperSubtask(
            "t2a-imagenet-c",
            "ImageNet-C transfer (15 corruptions, severity 3)",
            "paper_code/T2/Task2A/eval_imagenet_c.py",
            None,
            "Train on Gaussian only; largest gains on noise/frost/blur.",
        ),
    ),
    "t02b-chexpert-isotropic": (
        PaperSubtask(
            "t2b-pneumonia-clean",
            "Pneumonia chest X-ray — clean test (B0 vs E1 arms)",
            "paper_code/T2/Task2B/train.py",
            "t2b_chexpert_isotropic",
            "B0 best clean (91.7%); E1_no_pmh / E1 trade clean for shift robustness.",
        ),
        PaperSubtask(
            "t2b-robust-shift",
            "Robust eval — Gaussian + acquisition shifts",
            "paper_code/T2/Task2B/eval_robust.py",
            "t2b_chexpert_isotropic",
            "E1_no_pmh best worst-shift; E1 best saliency + embedding drift.",
        ),
        PaperSubtask(
            "t2b-saliency",
            "Saliency stability under noise",
            "paper_code/T2/Task2B/saliency_stability.py",
            None,
            "E1 (PMH) 0.723 vs B0 0.560 cosine stability.",
        ),
    ),
    "t03a-pose-gradient": (
        PaperSubtask(
            "t3a-calibrate-w",
            "Calibrate occlusion subspace W",
            "paper_code/T3/Task3A/calibrate_subspace.py",
            None,
            "Gradient-SVD on COCO pose features.",
        ),
        PaperSubtask(
            "t3a-train-arms",
            "Train baseline / E1 / E1_aniso / VAT",
            "paper_code/T3/Task3A/train.py",
            None,
            "E1_aniso +22.4 pp PCK vs baseline.",
        ),
        PaperSubtask(
            "t3a-eval-robust",
            "Robustness + embedding eval",
            "paper_code/T3/Task3A/eval.py",
            None,
            "Occlusion stress + drift metrics.",
        ),
    ),
    "t03b-depth-augmentation": (
        PaperSubtask(
            "t3b-pipeline",
            "NYU depth pipeline (train + calibrate)",
            "paper_code/T3/Task3B/run_pipeline.py",
            "t3b_depth_d3",
            "E1_aniso wins photometric hard stress.",
        ),
        PaperSubtask(
            "t3b-calibrate",
            "Photometric subspace calibration",
            "paper_code/T3/Task3B/calibrate_subspace.py",
            "t3b_depth_d3",
            "Aug-delta Gram rank 32.",
        ),
        PaperSubtask(
            "t3b-wrong-w",
            "E1_wrong control (random W)",
            "paper_code/T3/Task3B/run_strengthening.py",
            None,
            "Strengthening / falsification arm.",
        ),
    ),
    "t04a-vision-domain": (
        PaperSubtask(
            "t4a-domainnet",
            "DomainNet real→sketch",
            "paper_code/T4/Task4A/run_pipeline.py",
            "t4_domain_d4",
            "E1_multiscale +3.31 pp test acc.",
        ),
        PaperSubtask(
            "t4a-tdi",
            "Per-layer TDI geometry",
            "paper_code/T4/Task4A/tdi.py",
            "t4_domain_d4",
            "Domain Gram on hook features.",
        ),
        PaperSubtask(
            "t4a-train",
            "B0 / E1 / E1_multiscale training",
            "paper_code/T4/Task4A/train.py",
            "t4_domain_d4",
            "",
        ),
    ),
    "t04b-multilayer-vision": (
        PaperSubtask(
            "t4b-rare5",
            "Cityscapes rare-5 mIoU",
            "paper_code/T4/Task4B/run_rare5_all.py",
            "t4_domain_d4",
            "E1_multiscale +11.1 pp mIoU.",
        ),
        PaperSubtask(
            "t4b-subset",
            "Build rare-5 training subset",
            "paper_code/T4/Task4B/build_subset.py",
            None,
            "",
        ),
        PaperSubtask(
            "t4b-tdi",
            "Pixel-aligned per-layer TDI",
            "paper_code/T4/Task4B/tdi.py",
            "t4_domain_d4",
            "",
        ),
    ),
    "t05a-qm9-molecule": (
        PaperSubtask(
            "t5a-pipeline",
            "QM9 train + noise + embedding eval",
            "paper_code/T5/Task5A/run_pipeline.py",
            "t5_compositional_d5",
            "E1 clean MAE 24.921.",
        ),
        PaperSubtask(
            "t5a-train",
            "MolGCN training",
            "paper_code/T5/Task5A/train.py",
            "t5_compositional_d5",
            "",
        ),
        PaperSubtask(
            "t5a-noise",
            "Position-noise eval sweep",
            "paper_code/T5/Task5A/eval_noise.py",
            "t5_compositional_d5",
            "",
        ),
    ),
    "t05b-code-tokens": (
        PaperSubtask(
            "t5b-pipeline",
            "Clone detection train + eval",
            "paper_code/T5/Task5B/run_pipeline.py",
            "t5_compositional_d5",
            "E1 rename_bacc 0.9383.",
        ),
        PaperSubtask(
            "t5b-train",
            "CodeBERT clone training",
            "paper_code/T5/Task5B/train.py",
            "t5_compositional_d5",
            "",
        ),
        PaperSubtask(
            "t5b-eval",
            "Rename / reformat eval suites",
            "paper_code/T5/Task5B/eval.py",
            "t5_compositional_d5",
            "",
        ),
    ),
    "t06a-speech-whisper": (
        PaperSubtask(
            "t6a-run",
            "LibriSpeech four arms + WER",
            "paper_code/T6/task6A/run_task6a.py",
            None,
            "pmh_content_residual WER 14.63%.",
        ),
        PaperSubtask(
            "t6a-collect-w",
            "Content-residual subspace",
            "paper_code/T6/task6A/collect_W.py",
            None,
            "Only matched W fixes geometry.",
        ),
        PaperSubtask(
            "t6a-export",
            "Strengthening analysis JSON",
            "paper_code/T6/task6A/export_analysis.py",
            None,
            "",
        ),
    ),
    "t06b-temporal-har": (
        PaperSubtask(
            "t6b-multi-seed",
            "HAR multi-seed paper runs",
            "paper_code/T6/task6B/run_multi_seed.py",
            "t6_temporal_d6",
            "PMH 0.4099 vs 0.2794 @ stress 3.",
        ),
        PaperSubtask(
            "t6b-collect-w",
            "Collect W from baseline",
            "paper_code/T6/task6B/collect_W.py",
            "t6_temporal_d6",
            "",
        ),
        PaperSubtask(
            "t6b-stress",
            "Stress robustness eval",
            "paper_code/T6/task6B/stress_probe.py",
            "t6_temporal_d6",
            "",
        ),
    ),
    "t07a-llm-style": (
        PaperSubtask(
            "t7a-rm-eval",
            "RM behavioral eval (TQA n=500)",
            "paper_code/T7/task7A/evaluate_7a_behavioral.py",
            "t7a_style_d7",
            "Matched sycophancy 13.5%.",
        ),
        PaperSubtask(
            "t7a-dpo",
            "Geometric DPO + style geometry",
            "paper_code/T7/task7A/train_geometric_dpo.py",
            "t7a_style_d7",
            "margin_pmh Style TDI 1.836.",
        ),
        PaperSubtask(
            "t7a-pipeline",
            "Synthetic alignment pipeline",
            "paper_code/T7/task7A/run_task7a_pipeline.py",
            "t7a_style_d7",
            "",
        ),
    ),
    "t07b-adversarial-pgd": (
        PaperSubtask(
            "t7b-train",
            "CIFAR ViT PGD arms (seed 7)",
            "paper_code/T7/task7B/run_task7b.py",
            "t7b_pgd_d7",
            "pmh_aniso TDI 0.878.",
        ),
        PaperSubtask(
            "t7b-eval",
            "Adversarial + geometry eval",
            "paper_code/T7/task7B/run_task7b.py",
            "t7b_pgd_d7",
            "Correct W +8.6 pp PGD@4 vs wrong_W.",
        ),
        PaperSubtask(
            "t7b-summary",
            "Bootstrap CI summary",
            "paper_code/T7/task7B/run_task7b.py",
            None,
            "",
        ),
    ),
}


def _tasks() -> tuple[PaperTask, ...]:
    def _p(
        task_id: str,
        block: str,
        title: str,
        lemma: str,
        what_changes: str,
        stack: str,
        preset: str | None,
        install: str,
        not_for: str,
        demo_script: str | None = None,
    ) -> PaperTask:
        return PaperTask(
            task_id=task_id,
            block=block,
            title=title,
            lemma=lemma,
            what_changes=what_changes,
            stack=stack,
            preset=preset,
            install=install,
            not_for=not_for,
            page=f"docs/tasks/{task_id}.md",
            notebook=f"notebooks/tasks/{task_id}.ipynb",
            demo_script=demo_script,
        )

    return (
        _p(
            "t01-classical",
            "T1",
            "Classical ML + matched projection (ridge, SVM, k-NN, logistic)",
            "D1",
            "Frozen features; matched subspace W; classical classifiers — see FINAL.md battery.",
            "sklearn",
            "t1_office31_sklearn",
            'pip install "matching-pmh[sklearn,vision]"',
            "Severe nonlinear shift with no feature map; use deep PMH (T3–T7) instead.",
            demo_script="scripts/demos/office31_sklearn.py",
        ),
        _p(
            "t02a-vit-isotropic",
            "T2A",
            "ViT / image classifier — isotropic sensor noise",
            "D2",
            "Small **sensor / embedding noise**; class semantics unchanged.",
            "pytorch",
            "t2a_vit_isotropic",
            "pip install matching-pmh torch",
            "Large domain shift without D2 setup — use T4 instead.",
        ),
        _p(
            "t02b-chexpert-isotropic",
            "T2B",
            "Medical imaging — hospital / scanner embedding shift",
            "D2",
            "Hospital or scanner changes **appearance**; disease label unchanged.",
            "pytorch",
            "t2b_chexpert_isotropic",
            "pip install matching-pmh torch",
            "New disease labels at deploy only.",
        ),
        _p(
            "t03a-pose-gradient",
            "T3A",
            "Pose / keypoints — camera & studio shift",
            "D3",
            "Camera, lighting, viewpoint change; **same keypoint indices**.",
            "pytorch",
            None,
            "pip install matching-pmh torch",
            "Different skeleton or keypoint definitions at deploy.",
        ),
        _p(
            "t03b-depth-augmentation",
            "T3B",
            "Depth estimation — photometric shift",
            "D3",
            "Lighting/texture shift; **depth target meaning** unchanged.",
            "pytorch",
            "t3b_depth_d3",
            "pip install matching-pmh torch",
            "Different depth semantics or scale definition at deploy.",
        ),
        _p(
            "t04a-vision-domain",
            "T4A",
            "Vision domain shift (single-layer / ResNet)",
            "D4",
            "New camera, site, or geography; **same classes**.",
            "pytorch",
            "t4_domain_d4",
            "pip install matching-pmh torch",
            "New classes at deploy without relabeling.",
        ),
        _p(
            "t04b-multilayer-vision",
            "T4B",
            "Vision domain shift (multilayer FPN / U-Net)",
            "D4",
            "Texture **and** scene style shift together; same label map.",
            "pytorch",
            "t4_domain_d4",
            "pip install matching-pmh torch",
            "Single-layer hook enough — try T4A first.",
        ),
        _p(
            "t05a-qm9-molecule",
            "T5A",
            "Molecules / graphs (QM9-style)",
            "D5",
            "Conformer / position blocks move; **property label** fixed.",
            "pytorch",
            "t5_compositional_d5",
            "pip install matching-pmh torch",
            "Property definition changes at deploy.",
        ),
        _p(
            "t05b-code-tokens",
            "T5B",
            "Code models — token-group shift",
            "D5",
            "Imports/comments/identifiers change; **downstream label** fixed.",
            "pytorch",
            "t5_compositional_d5",
            'pip install "matching-pmh[hf]"',
            "New task or label at deploy.",
        ),
        _p(
            "t06a-speech-whisper",
            "T6A",
            "Speech / ASR — mic & room shift",
            "D6",
            "Microphone, room, codec; **transcript / word label** fixed.",
            "pytorch",
            None,
            "pip install matching-pmh torch",
            "Language or vocabulary change at deploy.",
        ),
        _p(
            "t06b-temporal-har",
            "T6B",
            "Time-series / HAR — sensor drift",
            "D6",
            "Sensor aging, device, session drift; **activity label** fixed.",
            "pytorch",
            "t6_temporal_d6",
            "pip install matching-pmh torch",
            "New activities only at deploy.",
        ),
        _p(
            "t07a-llm-style",
            "T7A",
            "LLM — format / tone / template",
            "D7",
            "Same facts, different **surface form** (JSON, bullets, tone).",
            "hf",
            "t7a_style_d7",
            'pip install "matching-pmh[hf]"',
            "Factual drift or new knowledge at deploy.",
        ),
        _p(
            "t07b-adversarial-pgd",
            "T7B",
            "Adversarial / PGD perturbations",
            "D7",
            "Small **input perturbations** are the production threat.",
            "pytorch",
            "t7b_pgd_d7",
            "pip install matching-pmh torch",
            "Unbounded arbitrary shift with no perturbation model.",
        ),
    )


def list_paper_tasks() -> list[PaperTask]:
    cat = {t.task_id: t for t in _tasks()}
    return [cat[tid] for tid in PAPER_TASK_IDS if tid in cat]


def get_paper_task(task_id: str) -> PaperTask:
    for t in list_paper_tasks():
        if t.task_id == task_id:
            return t
    raise KeyError(f"unknown paper task {task_id!r}")
