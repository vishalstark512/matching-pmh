# Publishing `matching-pmh` on PyPI (production)

**PyPI name:** `matching-pmh` — https://pypi.org/project/matching-pmh/  
**Import:** `import pmh`  
**CLI:** `pmh-train`

After a successful release, anyone can install with:

```bash
pip install matching-pmh
pip install "matching-pmh[hf]"      # Hugging Face D7 / Trainer
pip install "matching-pmh[hf-lora]" # + PEFT for example 11
```

---

## One-time setup (≈15 minutes)

### 1. Accounts

1. [pypi.org](https://pypi.org/account/register/) — production  
2. [test.pypi.org](https://test.pypi.org/account/register/) — optional dry run  
3. [github.com](https://github.com) — source + trusted publishing  

### 2. Reserve the project name on PyPI

On [pypi.org/manage/projects/](https://pypi.org/manage/projects/), the **first successful upload** of `matching-pmh` claims the name. Upload once manually or via CI (below).

### 3. Trusted publishing (recommended — no long-lived API token in GitHub)

**On PyPI** → [matching-pmh → Publishing](https://pypi.org/manage/project/matching-pmh/settings/publishing/) (or Account → Publishing) → **Add a new pending publisher**:

| Field | Value (must match CI claims exactly) |
|--------|--------|
| PyPI project name | `matching-pmh` |
| Owner | `vishalstark512` |
| Repository name | `matching-pmh` |
| Workflow name | `ci.yml` |
| Environment name | *(leave completely empty)* |

GitHub’s token claims include `environment: MISSING` — if you type `pypi` or anything here, publish will fail with `invalid-publisher`.

**Not** `publish-pypi.yml` — the publish job lives in **`ci.yml`** only.

Repeat for **TestPyPI** if you use `publish-testpypi.yml` (workflow name `publish-testpypi.yml`).

#### `invalid-publisher` troubleshooting

| Symptom | Fix |
|---------|-----|
| `valid token, but no corresponding publisher` | Add the publisher row above on the **matching-pmh** project, or delete a stale publisher with wrong workflow/repo. |
| Publisher on wrong workflow | Workflow must be `ci.yml`, not `publish-pypi.yml`. |
| Environment mismatch | Clear **Environment name** on PyPI (workflow does not use a GitHub Environment). |
| Project does not exist yet | One manual `twine upload` first (Path A), then add trusted publisher. |

### 4. GitHub repository

```powershell
cd C:\Users\Eigenaar\Desktop\matching-pmh
gh auth login
gh repo create matching-pmh --public --source=. --remote=origin
git add .
git commit -m "Release matching-pmh 0.7.0: walkthroughs, lab docs, example CI"
git branch -M main
git push -u origin main
```

Update `pyproject.toml` and `CITATION.cff` `[project.urls]` / `repository-code` if your URL is not `github.com/matching-pmh/matching-pmh`.

---

## Path A — Publish from your machine (first time, fastest)

```powershell
cd C:\Users\Eigenaar\Desktop\matching-pmh
pip install build twine
python -m build
twine check dist/*

# Optional dry run on TestPyPI first:
# twine upload --repository testpypi dist/*
# pip install -i https://test.pypi.org/simple/ matching-pmh==0.6.0

# Production PyPI (you will be prompted for API token; use scope "Entire account" or project-scoped):
twine upload dist/*
```

Create an API token on your [account settings](https://pypi.org/manage/account/) page → **API tokens** → **Add API token** (scope: entire account for first upload, or project **matching-pmh** after it exists).

Verify:

```powershell
pip install matching-pmh==0.7.0
pmh-train list-methods
python -c "import pmh; print(pmh.__version__)"
```

---

## Path B — Publish via GitHub tag (after trusted publishing is configured)

```powershell
git tag v0.7.0
git push origin v0.7.0
```

Workflow `.github/workflows/ci.yml` runs tests, builds, and publishes to **pypi.org** on tags `v*`.

---

## Version bump checklist (every release)

1. `pyproject.toml` → `version`
2. `src/pmh/__init__.py` → `__version__`
3. `CITATION.cff` → `version`
4. `CHANGELOG.md` → new section
5. `git tag vX.Y.Z && git push origin vX.Y.Z`

---

## Fallback: API token in GitHub Actions (fastest if trusted publishing is blocked)

1. PyPI → Account → **API tokens** → Add token (scope: project `matching-pmh` or entire account).
2. GitHub → `vishalstark512/matching-pmh` → **Settings → Secrets and variables → Actions** → New secret:
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-…` (full token, once)
3. Re-run the failed **publish** job, or re-push the tag:

```powershell
git push origin :refs/tags/v2.0.0
git tag -a v2.0.0 -m "Release matching-pmh 2.0.0"
git push origin v2.0.0
```

When `PYPI_API_TOKEN` is set, CI uses the token instead of OIDC.

---

## GitHub Pages (docs site)

**URL:** https://vishalstark512.github.io/matching-pmh/

Workflow: `.github/workflows/docs.yml` (build on PR; deploy on `main` push).

If deploy fails with `Not Found`, Pages is not enabled for Actions yet:

1. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**
2. Re-run the **docs** workflow

Details: [docs/GITHUB_PAGES_SETUP.md](docs/GITHUB_PAGES_SETUP.md)

---

## Paper cross-link

Set in `submission_grand_unification/macros.tex`:

```latex
\MatchingPmhRepoUrl{https://github.com/YOUR_USER/matching-pmh}
```

Add to reproducibility text: `pip install matching-pmh` (PyPI).

---

## Local preflight (before any upload)

```powershell
pip install build twine
python -m build
twine check dist/*
pip install dist\matching_pmh-0.6.0-py3-none-any.whl
pmh-train list-methods
pytest -q
```

---

## Notes

- **Torch** is a core dependency; wheels are large — that is normal for ML libraries on PyPI.
- The name `pmh` on PyPI may be taken by unrelated projects; this package is **`matching-pmh`** only.
- Do not commit `dist/`, `artifacts/`, or API tokens.
