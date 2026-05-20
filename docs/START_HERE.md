# Start here (three gates)

**Main guide:** **[Find your application](APPLICATIONS.md)** — decision tree, nuisances in plain English, 7-step walkthroughs.

This page is only the **go / no-go** checklist. Then [Golden paths](GOLDEN_PATHS.md) (one code section).

[Full site map](MAP.md)

```bash
pip install matching-pmh
pmh-train route --search pose    # keyword → matching applications
pmh-train route --task pose_or_keypoints
```

---

## Three gates (if unsure)

| # | Question | If **no** |
|---|----------|-----------|
| 1 | **Same labels** on A and B? (same keypoints, classes, clinical definition) | PMH is wrong tool — label shift |
| 2 | **Some data from site B**? (even unlabeled) | Collect deploy data (or LLM style pairs) |
| 3 | Hook a **representation** `h`? | [G2](GOLDEN_PATHS.md#g2) frozen features first |

```python
from pmh import check_applicability, explain_task
print(explain_task("pose_or_keypoints"))   # full walkthrough in terminal
```

---

## Quick links

| I am doing… | Go to |
|-------------|--------|
| **Any task — find myself in the list** | [APPLICATIONS.md](APPLICATIONS.md) |
| Copy-paste code | [GOLDEN_PATHS.md](GOLDEN_PATHS.md) |
| What to read / skip | [MAP.md](MAP.md) |
| Honest expectations | [WHEN_PMH_HELPS.md](WHEN_PMH_HELPS.md) |
| Errors | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |

```bash
pmh-train wizard
pmh-train doctor
```
