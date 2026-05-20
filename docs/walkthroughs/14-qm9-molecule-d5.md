# Walkthrough 14: QM9 / molecules (D5) — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g4) · **Route:** `pmh-train route --task compositional_coordinates` · **Step 5:** paper T5A controls
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Estimator** | D5 compositional on molecular coordinates |
| **Script** | `examples/16_qm9_molecule_d5.py` |

[Walkthrough 5](05-compositional-d5.md)

---

## Who this is for

Molecular property prediction where **shift-related coordinates** (e.g. conformer / environment block) are known separate from task atoms.

---

## Your deployment shift sentence

*“Solvent / conformer block shifts; property label from solute unchanged.”*

---

## Step-by-step

1. Define `nuisance_indices` on your feature vector.
2. Run `examples/16_qm9_molecule_d5.py` for wiring.
3. `PMHTrainer(..., nuisance_indices=...)`.

```bash
python examples/16_qm9_molecule_d5.py
```

---

## Adaptation worksheet

| Example | Your GNN / fingerprint |
|---------|------------------------|
| Index split | Your chemistry semantics |

---

## Next steps

- [5 — D5](05-compositional-d5.md)
- [hooks.md](../INTEGRATE.md) — `encoder_gnn_mean_pool`
