"""Synthesized paper block outcomes (from ``paper_code`` / FINAL.md — not library demos)."""

from __future__ import annotations

from dataclasses import dataclass

from pmh.paper_tasks import PAPER_TASK_IDS, get_paper_task


@dataclass(frozen=True)
class PaperBlockFinding:
    task_id: str
    block: str
    title: str
    lemma: str
    headline: str
    status: str  # pass | partial | documented_failure
    final_path: str
    stack: str


# Headlines aligned with scripts/render_handcrafted_tasks DOC_EXTRA + T1 narrative
_HEADLINES: dict[str, tuple[str, str, str]] = {
    "t01-classical": (
        "Ridge theorem + oracle-W on MNIST/Fashion/SVHN; Office-31: CORAL > PMH on frozen ResNet, "
        "PMH > B0 on SVM — **documented D1 eigengap** case.",
        "partial",
        "paper_code/T1/classical_pmh/FINAL.md",
    ),
    "t02a-vit-isotropic": (
        "ViT-B/16 isotropic PMH: **+4.29 pp** mean ImageNet-C; TDI **−58%** at σ=0.10.",
        "pass",
        "paper_code/T2/Task2A/FINAL.md",
    ),
    "t02b-chexpert-isotropic": (
        "CheXpert E1: best saliency **0.723**; ~**9×** lower embedding drift vs baseline.",
        "pass",
        "paper_code/T2/Task2B/FINAL.md",
    ),
    "t03a-pose-gradient": (
        "COCO pose E1_aniso: **54.49%** PCK@0.05 (+22.4 pp vs baseline 32.07%).",
        "pass",
        "paper_code/T3/Task3A/FINAL.md",
    ),
    "t03b-depth-augmentation": (
        "Depth photometric hard stress: E1_aniso AbsRel **0.2152** (wins on combined_hard).",
        "pass",
        "paper_code/T3/Task3B/FINAL.md",
    ),
    "t04a-vision-domain": (
        "DomainNet real→sketch E1_multiscale: **42.15%** acc (+3.31 pp vs B0 38.84%).",
        "pass",
        "paper_code/T4/Task4A/FINAL.md",
    ),
    "t04b-multilayer-vision": (
        "GTA5→Cityscapes rare-5 mIoU **30.75%** (+11.1 pp vs B0 19.68%).",
        "pass",
        "paper_code/T4/Task4B/FINAL.md",
    ),
    "t05a-qm9-molecule": (
        "QM9 position PMH: clean MAE **24.921**; robust under σ=0.2 Å noise.",
        "pass",
        "paper_code/T5/Task5A/FINAL.md",
    ),
    "t05b-code-tokens": (
        "Code rename stress: E1 rename_bacc_ratio **0.9383** vs B0 **0.8297**; wrong blocks fail.",
        "pass",
        "paper_code/T5/Task5B/FINAL.md",
    ),
    "t06a-speech-whisper": (
        "Whisper/Libri content-residual: other-WER **14.63%** (−8.6 pp vs 23.26%).",
        "pass",
        "paper_code/T6/task6A/FINAL.md",
    ),
    "t06b-temporal-har": (
        "HAR stress 3.0: balanced acc **0.4099** vs baseline **0.2794** (3 seeds).",
        "pass",
        "paper_code/T6/task6B/FINAL.md",
    ),
    "t07a-llm-style": (
        "Style RM + DPO: sycophancy **38.5%→13.5%**; margin_pmh Style TDI **1.836**.",
        "pass",
        "paper_code/T7/task7A/FINAL.md",
    ),
    "t07b-adversarial-pgd": (
        "CIFAR PGD-W pmh_aniso: TDI **0.878** (−19% vs 1.090); clean **80.9%**.",
        "pass",
        "paper_code/T7/task7B/FINAL.md",
    ),
}


def list_paper_findings() -> list[PaperBlockFinding]:
    out: list[PaperBlockFinding] = []
    for tid in PAPER_TASK_IDS:
        t = get_paper_task(tid)
        h, status, final = _HEADLINES.get(
            tid,
            (t.what_changes, "pass", f"paper_code/{t.block}/"),
        )
        out.append(
            PaperBlockFinding(
                task_id=tid,
                block=t.block,
                title=t.title,
                lemma=t.lemma,
                headline=h,
                status=status,
                final_path=final,
                stack=t.stack,
            )
        )
    return out


def synthesis_paragraphs() -> list[str]:
    """Overall paper narrative (prose, for HTML/markdown)."""
    return [
        "The Perturbation Matching Hypothesis (PMH) treats label-preserving deploy change as one "
        "estimation problem: learn the geometry of nuisance variation, train with a matched penalty, "
        "and falsify with wrong-direction and isotropic controls before claiming deploy gains.",
        "**12 of 13** pre-registered blocks meet their pass criteria in block-specific "
        "`paper_code` reproduction scripts (see each `FINAL.md`). Wins span classical projection, "
        "ViT noise robustness, pose and depth, domain adaptation (DomainNet, Cityscapes), molecules, "
        "code renames, speech, HAR, LLM style, and PGD robustness.",
        "**T1 / Office-31** is the honest partial case: on frozen ResNet-18 features, CORAL can beat "
        "projection-only PMH on accuracy; PMH still beats ERM and wrong-W controls — illustrating "
        "Lemma D1 eigengap limits, not a silent library bug.",
        "Falsification arms (matched vs wrong-W vs isotropic) recur across blocks: gains tied to "
        "estimated nuisance geometry, not generic regularization.",
    ]
