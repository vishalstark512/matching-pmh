# Glossary (practitioner ↔ paper)

Use this page in the CLI, notebooks, and reports. Theory notation stays in [PRINCIPLE.md](PRINCIPLE.md) and [`main.pdf`](../main.pdf).

| Practitioner term | Research / code | Meaning |
|-------------------|-----------------|--------|
| Train site A / deploy site B | source / target domain | Where you fit vs where you score |
| Deployment shift covariance | $\Sigma_{\text{task}}$ | Covariance of label-preserving representation change |
| Shift-matched training penalty | matched PMH | Train so Jacobian energy follows estimated deploy geometry |
| Wrong-direction control | wrong-W arm | Deliberately misaligned subspace (Step 5 falsification) |
| Generic isotropic control | isotropic arm | Spherical penalty, not matched to your shift |
| New camera or site | `nuisance="domain_shift"` (D4) | Unlabeled target batches OK |
| Labels on both sites | `nuisance="subspace"` (D1) | Class-conditional cross-domain subspace |
| Known aug list | `nuisance="augmentation"` (D3) | Finite transforms you can enumerate |
| Format / tone shift | `nuisance="style"` (D7) | Same content, different surface (LLM) |
| Geometry check | preflight | Is the estimated shift identifiable? (not accuracy) |
| Step 5 falsification | Lemma C protocol | Matched must beat wrong + isotropic on deploy holdout |
| PMH / task ratio | — | PMH loss ÷ task loss; **target 5--30%**, hard cap at `pmh_max_task_ratio` |
| Ship / do not ship | Step 5 pass/fail | `report.ship_verdict()` |

---

## Loss scale (most common training issue)

| Setting | Role |
|---------|------|
| `PMHConfig.pmh_max_task_ratio` | **Hard cap** — PMH term ≤ this × task loss (default **0.30** = 30%) |
| `PMHConfig.pmh_min_task_ratio` | **Warn** if PMH falls below this × task (default **0.05** = 5%) |
| `PMHConfig.weight` | Scales raw penalty before cap |
| `PMHConfig.golden_path()` | Balanced defaults + task-ratio cap |

Details: [LOSS_SCALING.md](LOSS_SCALING.md)
