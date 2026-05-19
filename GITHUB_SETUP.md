# Publish `matching-pmh` on GitHub

The library lives only in this folder (`Desktop/matching-pmh`), not inside the paper repo.

## 1. Create the remote repository

On GitHub: **New repository** → name `matching-pmh` → public → **no** README (this tree has one).

Or install [GitHub CLI](https://cli.github.com/) and run:

```powershell
cd C:\Users\Eigenaar\Desktop\matching-pmh
gh auth login
gh repo create matching-pmh --public --source=. --remote=origin
```

## 2. Publish on PyPI (full package)

See **`PUBLISHING.md`** for the complete guide. Summary:

1. Build: `python -m build` && `twine check dist/*`
2. Upload: `twine upload dist/*` (or tag `v0.6.0` + trusted publishing on GitHub)
3. Users install: `pip install matching-pmh`

The name `matching-pmh` is **not yet on PyPI** until you upload once.

## 3. First commit and push

```powershell
cd C:\Users\Eigenaar\Desktop\matching-pmh
git init
git add .
git commit -m "Initial release: estimate_sigma_task (D1-D7) and PMH penalties (v0.1.0)"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/matching-pmh.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub account. Update `CITATION.cff` and `pyproject.toml` `[project.urls]` if the URL differs.

## 4. Paper cross-link

After the repo exists, set the URL in the manuscript macro (once):

`submission_grand_unification/macros.tex` → `\MatchingPmhRepoUrl{https://github.com/YOUR_USERNAME/matching-pmh}`

Then rebuild the PDF (`pdflatex` ×2 in `submission_grand_unification/`).

## 4. Remove duplicate from paper tree

Delete the old copy under `Paper2/pmh/` (only a stub README should remain there).

## 5. Optional: PyPI

When ready for `pip install matching-pmh`:

```powershell
pip install build twine
python -m build
twine upload dist/*
```
