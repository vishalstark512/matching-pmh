# Scripts

| Script | Purpose |
|--------|---------|
| [render_handcrafted_tasks.py](render_handcrafted_tasks.py) | Regenerate `docs/tasks/*.md` (except handcrafted) + `notebooks/tasks/*.ipynb` |
| [build_paper_docs.py](build_paper_docs.py) | Regenerate `docs/tasks/index.md` only |
| [download_office31.py](download_office31.py) | Download Office-31 for T1 (local path, not committed) |
| [generate_reference_benchmark.py](generate_reference_benchmark.py) | Optional sklearn benchmark JSON for maintainers |
| [demos/](demos/README.md) | Optional CLI: first-run smoke, Office-31 sklearn, benchmark table |
| [configs/](configs/) | Sample `pmh-train` JSON configs |

```bash
python scripts/render_handcrafted_tasks.py
python scripts/build_paper_docs.py   # optional: refresh index only
PMH_QUICK=1 python scripts/demos/first_run_domain_shift.py
pmh-train doctor
pmh-train evaluate --demo
pmh-train route --list
```

CLI is intentionally thin (`doctor`, `evaluate`, `route`). Batch estimate/benchmark JSON jobs: use Python API or `scripts/configs/` with your own driver.

PowerShell release helpers: `preflight_release.ps1`, `publish_github.ps1`, `upload_pypi.ps1`
