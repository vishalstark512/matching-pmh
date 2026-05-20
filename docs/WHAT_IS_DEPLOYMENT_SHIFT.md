# What is a “deployment shift”? (and what `nuisance=` means)

**You do not need the word “nuisance” to use this library.**

In the paper, *nuisance* means: **ways your inputs can change at deploy time while the label stays the same.**  
In code, that idea is just the string you pass as **`nuisance=`** — a **shift type** picker, not a moral judgment about your data.

---

## Plain English

| Paper / code word | Say this instead |
|-------------------|------------------|
| Nuisance \(n\) | **Deploy shift** — camera, hospital, mic, wording, time, etc. |
| \(\Sigma_{\mathrm{task}}\) | **Geometry of deploy shifts** (estimated once) |
| `nuisance="domain_shift"` | **“Site A vs site B looks different, same labels”** (default for most vision/tabular) |
| `nuisance="subspace"` | **“Same classes, labeled on both sites”** |
| `nuisance="style"` | **“LLM formatting differs, facts the same”** |
| D1–D7 | **Evidence IDs** — ignore until you need the appendix |

---

## Default (80% of users)

**Hospital A → Hospital B**, or **warehouse camera → store camera**, or **lab A cohort → lab B cohort**:

- Same disease / product class / intent  
- Images, audio, or rows *look* different  
- You have batches from both sites (target labels optional for estimate)

```python
nuisance="domain_shift"   # API name — means cross-site look, not new classes
```

```bash
pmh-train route --task vision_classification
pmh-train shifts    # print all shift types in plain English
```

---

## Pick from what you notice

```python
from pmh import format_shift_types, suggest_nuisance

print(format_shift_types())
print(suggest_nuisance(has_target_domain=True, has_target_labels=False))
# -> nuisance='domain_shift' ...
```

| You notice | `nuisance=` | When |
|------------|-------------|------|
| Site / camera / cohort **look**, deploy unlabeled OK | `domain_shift` | Default |
| **Labels on both** sites, class geometry moves | `subspace` | Office-31 with labels on A and B |
| LLM **format** changes, facts fixed | `style` | Style-pair JSONL or corpora |
| Named aug modes (blur, crop, …) | `augmentation` | You generate the modes |
| Only **some coordinates** of features move | `compositional` | You know `nuisance_indices` |
| **Time** drift along sequences | `temporal` | Sequence data |
| No direction — sensor noise only | `isotropic` | Rare; specialist |

Full table + lemmas: [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) (appendix).

---

## What PMH is *not* for

- **New classes** at deploy that did not exist in training  
- **Label definition changes** (disease criteria, taxonomy, policy-only LLM rules)  
- “Make any model robust to everything” without a deploy story  

See [WHEN_PMH_HELPS.md](WHEN_PMH_HELPS.md).

---

## One sentence for your notebook

> We estimate how deployment can change inputs without changing labels, then train so the model is less sensitive along those directions, and we verify on a deploy holdout with falsification arms.

Next: [Five-step recipe](FIVE_STEP_RECIPE.md) · [Applications](APPLICATIONS.md)
