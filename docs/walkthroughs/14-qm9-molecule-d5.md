# Walkthrough 14: QM9 / molecules (D5) — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D5 compositional on molecular coordinates |
| **Script** | `examples/16_qm9_molecule_d5.py` |

[Walkthrough 5](05-compositional-d5.md)

---

## Who this is for

Molecular property prediction where **nuisance coordinates** (e.g. conformer / environment block) are known separate from task atoms.

---

## Your nuisance sentence

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
- [hooks.md](../hooks.md) — `encoder_gnn_mean_pool`
