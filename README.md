# matching-pmh

**Train on site A. Deploy on site B. Same labels.**

> **Deploy QA gate:** When your model fits training data but breaks on deploy — same task, same labels, different site, camera, or corpus — PMH estimates how representations should move at deploy, trains a **shift-matched** penalty, and tells you **ship** or **do not ship** only after matched beats wrong-direction and generic controls on deploy holdout. [Start here →](docs/START.md)

This repository ships the **Perturbation Matching Hypothesis (PMH)** — a geometric theory of how training losses should respond to *label-preserving* deployment change. The paper ([`main.pdf`](main.pdf)) argues that domain shift, sensor noise, augmentation stress, compositional drift, temporal drift, style, and classical anisotropic penalties are **one statistical problem**: estimate the deployment nuisance covariance $\Sigma_{\text{task}}$, then train so the encoder Jacobian is matched to that geometry. CORAL, adversarial training, augmentation, metric learning, and alignment constraints become **different estimators of the same object**, not unrelated “robustness tricks.”

**matching-pmh** is the library + thirteen worked demos that implement the paper’s **five-step recipe** on real stacks (PyTorch, sklearn, Hugging Face). You are not picking a regularizer off a menu; you are identifying $\Sigma_{\text{task}}$ for *your* deploy shift and applying the matched PMH loss from Eq. (4) in the paper.

---

## The idea in plain language

**What moves at deploy without changing the label?**  
Examples: new camera or hospital (vision), new microphone (speech), new writing style (LLM), known lighting aug (depth), PGD-like perturbations (security). All of these are instances of a single random displacement $n$ with covariance $\Sigma_{\text{task}} = \mathrm{Cov}(n)$.

**What training should do.**  
Add a PMH term that penalizes encoder Jacobian energy along a matrix $\Sigma'$ whose **column space covers** the nuisance range. When $\Sigma'$ is **matched** to $\Sigma_{\text{task}}$, deployment drift in representations can be driven down; when $\Sigma'$ is **isotropic** or **wrong**, the theory predicts specific failure modes — and the library runs those arms as **controls**, not optional extras.

**What makes this a theory, not a hack.**  
The paper proves range coverage is necessary for quadratic Jacobian penalties, gives matched sufficiency in the linear model, extends to deep global minima under stated assumptions, and supplies falsification lemmas (wrong subspace, signal-aligned penalty) tested before you trust a deploy gain. See [`main.pdf`](main.pdf) §2–5 and [block findings](docs/findings.html).

---

## The five-step recipe (product spine)

Same steps in every notebook (§1–§8) and in `pmh.recipe`:

| Step | Question you answer | Library entry points |
|------|---------------------|----------------------|
| **0 — Scope** | Same label semantics on train (A) and deploy (B)? | `check_applicability()` |
| **1 — Identify** | Which nuisance family fits? (seven types, D1–D7) | `suggest_nuisance()` · [task table below](#find-your-deployment-story-t1-through-t7) |
| **2 — Estimate** | $\hat{\Sigma}_{\text{task}}$ from your data | `PMHTrainer.estimate()` · `PMHMatcher.fit()` · `estimate_style_sigma()` |
| **3 — Apply** | Matched PMH on hook $h$ (train) or projection (frozen features) | `PMHTrainer.fit` · `robust_fit` · `PMHLoss` |
| **4 — Protocol** | Keep PMH at **5--30%** of task loss (hard cap) | `PMHConfig.golden_path()` · [LOSS_SCALING](docs/LOSS_SCALING.md) |
| **5 — Evidence** | Matched beats wrong-direction and isotropic on **deploy holdout** | `evaluate_robust_fit` · `evaluate_baseline_vs_pmh` |

```text
Scope → identify nuisance family → estimate $\Sigma_{\text{task}}$ → matched PMH train → falsify on deploy holdout.
```

Details: [Quickstart](docs/QUICKSTART.md) · [Will PMH help?](docs/WHEN_PMH_HELPS.md) · [API](docs/api/index.md)

---

## What this repo promises

| We provide | We do not claim |
|------------|-----------------|
| A **closed, falsifiable** training recipe once $\Sigma_{\text{task}}$ is identified | Universality on every leaderboard |
| **13 pre-registered blocks** (T1–T7) as copy-paste playbooks | That matched PMH always beats CORAL, DANN, or PGD-AT |
| Built-in **matched / wrong / isotropic** arms (Lemma C, Cor. E in the paper) | PMH on label-changing shifts (e.g. spurious correlation) |
| Theory-aligned estimators D1–D7 + geometry probes (`tdi`, …) | One demo preset replaces your domain data or reproduces every paper table row without tuning |

Honest boundaries (from the paper): Colored MNIST / Waterbirds-style **label-correlated** nuisance is out of scope; Office-31 is a documented case where **estimator eigengap** can fail (Lemma D1), not a silent bug.

**Pre-registered evidence:** **12/13** paper blocks pass their criteria in [`main.pdf`](main.pdf) (see [findings.html](docs/findings.html)); Office-31 is the predicted **D1 failure** when the cross-domain subspace is ill-conditioned — run Step 5 before shipping.

### Paper numbers vs this library

Block accuracies, mIoU gains, and other **reported figures** in the README, task pages, and [`main.pdf`](main.pdf) tables are **paper results** — full benchmarks, datasets, and schedules described in the PDF.

The **`pmh` library** on PyPI is for **general use** on your stack: same estimators and five-step recipe, but **different** demo loaders, defaults, and integration paths. It will **not** automatically replicate those paper numbers out of the box. Expect **iteration** on your side — hook choice, rank, `PMHConfig` / loss scale ([LOSS_SCALING](docs/LOSS_SCALING.md)), more target data, and Step 5 on **your** deploy holdout — before you treat a run as “correct.” Notebooks under `notebooks/tasks/` teach the workflow on built-in demos.

Short theory spine (no PDF required to start): **[docs/PRINCIPLE.md](docs/PRINCIPLE.md)**.  
**Synthesized block outcomes (HTML):** [docs/findings.html](docs/findings.html) — regenerate with `python scripts/build_findings_html.py`.

---

## Choose your depth

| You want… | Open |
|-----------|------|
| Plain-language principle + five steps | [docs/PRINCIPLE.md](docs/PRINCIPLE.md) |
| “Will this help my deploy shift?” | [docs/WHEN_PMH_HELPS.md](docs/WHEN_PMH_HELPS.md) |
| Copy-paste task for your nuisance | [docs/tasks/index.md](docs/tasks/index.md) → notebook §8 |
| Sklearn / frozen embeddings (T1) | [t01-classical](docs/tasks/t01-classical.md) · `compare_arms_sklearn` |
| PyTorch site/camera (T4) | [t04a-vision-domain](docs/tasks/t04a-vision-domain.md) · `PMHTrainer` (class-aligned D4) |
| Per-layer domain Gram (T4B) | [t04b-multilayer-vision](docs/tasks/t04b-multilayer-vision.md) · `PMHTrainer(train_mode="feature_diff")` |
| Full proofs + block numbers | [`main.pdf`](main.pdf) · [findings.html](docs/findings.html) |
| Matched / wrong / isotropic benchmark | `run_benchmark_protocol` · `compare_arms` |

---

## Find your deployment story (T1 through T7)

Tasks are **examples** of the same principle — pick the closest **deploy change**, open the page + notebook, Run All on demo data, then plug in your pipeline in §8. Order follows the paper blocks (T1 first).

| Task | What changes at deploy (labels fixed) | Real situations like yours | How $\hat{\Sigma}_{\text{task}}$ is built | `nuisance=` | Start |
|------|--------------------------------------|----------------------------|-------------------------------------------|-------------|--------|
| **T1** | Embedding cloud shifts between sites | Office-31; two labs’ tabular features; frozen ResNet vectors | Cross-domain subspace on features (D1) | `subspace` | [T1](docs/tasks/t01-classical.md) |
| **T2A** | Undirected input corruption (no fixed direction) | ImageNet-C; sensor noise; blur/JPEG | Isotropic $\sigma^2 I$ (D2) | `isotropic` | [T2A](docs/tasks/t02a-vit-isotropic.md) |
| **T2B** | Scanner / site appearance on X-ray | Hospital drift on CheXpert-style data | Isotropic $\sigma$ (D2) | `isotropic` | [T2B](docs/tasks/t02b-chexpert-isotropic.md) |
| **T3A** | Camera / lighting; same keypoint semantics | Studio→wild pose; broadcast→fan video | Augmentation-induced deltas (D3) | `augmentation` | [T3A](docs/tasks/t03a-pose-gradient.md) |
| **T3B** | Photometry; depth meaning unchanged | Lighting on depth; synthetic→real RGB-D | Augmentation deltas (D3) | `augmentation` | [T3B](docs/tasks/t03b-depth-augmentation.md) |
| **T4A** | New visual domain; same classes | Photo→sketch; warehouse A→B; country shift | Source−target feature Gram (D4) | `domain_shift` | [T4A](docs/tasks/t04a-vision-domain.md) |
| **T4B** | Sim→real texture + layout; same seg map | GTA5→Cityscapes; synthetic seg→real | Domain Gram per layer (D4; paper multiscale) | `domain_shift` | [T4B](docs/tasks/t04b-multilayer-vision.md) |
| **T5A** | 3D atom coordinates move; property fixed | QM9 conformers; pose grids | Compositional blocks (D5) | `compositional` | [T5A](docs/tasks/t05a-qm9-molecule.md) |
| **T5B** | Token groups change; code label fixed | Renames; comment stripping | Nuisance indices on tokens (D5) | `compositional` | [T5B](docs/tasks/t05b-code-tokens.md) |
| **T6A** | Channel / room / codec; same transcript | New mic; Libri conditions | Temporal / content-residual (D6) | `temporal` | [T6A](docs/tasks/t06a-speech-whisper.md) |
| **T6B** | Sensor drift over time | HAR placement; IMU aging | Temporal residual (D6) | `temporal` | [T6B](docs/tasks/t06b-temporal-har.md) |
| **T7A** | Surface form; facts unchanged | Bulleted vs prose; tone shift in LLMs | Style pairs → Gram (D7) | `style` | [T7A](docs/tasks/t07a-llm-style.md) |
| **T7B** | Adversarial directions at deploy | PGD stress; spoof patches | PGD delta subspace (D7) | `style` / PGD doc | [T7B](docs/tasks/t07b-adversarial-pgd.md) |

Full index: **[13 tasks](docs/tasks/index.md)** · [notebooks](notebooks/README.md)

```bash
pmh-train route --list
```

### Seven nuisance types (one object, seven estimators)

| Type | $\Sigma_{\text{task}}$ is… | Data you typically need |
|------|---------------------------|-------------------------|
| **D1 subspace** | Low-rank cross-domain difference | Labeled source + target features |
| **D2 isotropic** | Spherical noise level | Train distribution (+ noise level if known) |
| **D3 augmentation** | Span of aug-induced feature moves | Train + known augmentations |
| **D4 domain** | Gram of **class-aligned** source−target diffs (labels optional) | Train + deploy batches (labeled pairs preferred) |
| **D5 compositional** | Covariance on named coordinates | Train + which dims are nuisance |
| **D6 temporal** | Drift along time / sequence | Trajectories, sensor series |
| **D7 style** | Style / attack direction covariance | Same-content pairs or PGD deltas |

If two rows sound similar, start with **T1** (frozen vectors) or **T4A** (end-to-end vision). Your benchmark name does not matter — the **nuisance law** does.

---

## Adapt any similar pipeline

1. Match **deploy change** to a row above (not the paper ID).  
2. Open that task’s notebook — sections **1–8** always follow the five-step recipe.  
3. Replace demo loaders with your data; keep the same `nuisance=` and estimate call.  
4. Run **Step 5** on deploy holdout; ship only if **matched** beats **wrong-direction** and **generic isotropic** (see [WHEN_PMH_HELPS](docs/WHEN_PMH_HELPS.md)).

The demos in `scripts/demos/` and `notebooks/tasks/` exist to show the **same ordering** the theory predicts (`matched` → `isotropic` → `wrong` on geometry and drift metrics), not to define thirteen separate products.

---

## Start here

**Practitioners:** [docs/START.md](docs/START.md) — one function, one ship verdict (no paper, auto shift type).

```bash
pip install matching-pmh torch
pip install "matching-pmh[sklearn]"   # frozen-feature path
pmh-train try --quick                  # ~1 min: train + deploy report + SHIP / DO NOT SHIP
```

```python
from pmh import try_pmh
from pmh.pytorch_eval import pytorch_demo_loaders

bundle = pytorch_demo_loaders(n=400, seed=0)
report = try_pmh(
    bundle.model, bundle.train_loader, bundle.val_loader,
    source_batches=bundle.source_batches, target_batches=bundle.target_batches,
    hook=bundle.encoder, head=bundle.head, epochs=5,
)
print(report.deploy_summary())
print(report.ship_verdict())  # auto nuisance= — you do not pick D1–D7 first
```

```bash
pmh-train doctor
pmh-train evaluate --demo --stack pytorch
pmh-train try --stack multilayer --quick   # T4B RGB CNN feature-diff demo
```

| Path | Notebook | When |
|------|----------|------|
| **T1** classical / frozen features | [t01-classical.ipynb](notebooks/tasks/t01-classical.ipynb) · [Colab](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t01-classical.ipynb) | sklearn, embeddings |
| **T4A** vision domain | [t04a-vision-domain.ipynb](notebooks/tasks/t04a-vision-domain.ipynb) · [Colab](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t04a-vision-domain.ipynb) | PyTorch site/camera |
| **T4B** multilayer vision | [t04b-multilayer-vision.ipynb](notebooks/tasks/t04b-multilayer-vision.ipynb) · [Colab](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t04b-multilayer-vision.ipynb) | Per-layer feature-diff PMH |

Read the theory: **[`main.pdf`](main.pdf)** · Block summary: **[findings.html](docs/findings.html)**

---

## Documentation map

| Doc | Role |
|-----|------|
| [`main.pdf`](main.pdf) | Full theory, theorems, thirteen blocks |
| [docs/START.md](docs/START.md) | **Golden path** — `try_pmh`, auto shift type, ship verdict |
| [docs/MIGRATE.md](docs/MIGRATE.md) | CORAL, sklearn, HF, augmentation |
| [docs/LOSS_SCALING.md](docs/LOSS_SCALING.md) | PMH vs task loss (5--30%, enforced cap) |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Plain language ↔ code |
| [docs/PRINCIPLE.md](docs/PRINCIPLE.md) | Short PMH spine ($\Sigma_{\text{task}}$, five steps, library vs paper) |
| [docs/index.md](docs/index.md) | Site hub |
| [docs/cookbook/](docs/cookbook/) | Lightning + HF integration sketches |
| [QUICKSTART.md](docs/QUICKSTART.md) | Install + commands |
| [tasks/index.md](docs/tasks/index.md) | All tasks T1–T7 + deploy table |
| [WHEN_PMH_HELPS.md](docs/WHEN_PMH_HELPS.md) | Fit, misfit, controls |
| [api/index.md](docs/api/index.md) | `PMHTrainer`, presets, evaluate |

---

## Links

[PyPI](https://pypi.org/project/matching-pmh/) · [Documentation site](https://vishalstark512.github.io/matching-pmh/) · [Contributing](CONTRIBUTING.md)
