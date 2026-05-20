# GitHub Pages setup (one time)

If **Deployments → github-pages** shows failed runs, the docs **build** usually succeeded but **deploy** could not create a Pages site. Fix this once in the repo settings.

## 1. Enable Pages with GitHub Actions

1. Open **https://github.com/vishalstark512/matching-pmh/settings/pages**
2. Under **Build and deployment** → **Source**, choose **GitHub Actions** (not “Deploy from a branch”).
3. Save. GitHub creates the `github-pages` environment automatically.

## 2. Re-run the docs workflow

1. **Actions** → **docs** → open the latest run → **Re-run all jobs**  
   Or: **Run workflow** (workflow_dispatch) on `main`.
2. Wait until **build** and **deploy** are both green.
3. Site URL: **https://vishalstark512.github.io/matching-pmh/** (can take 1–2 minutes after deploy).

## What the workflow does

- **build** — `mkdocs build`, upload `site/` as a Pages artifact  
- **deploy** — `actions/deploy-pages` (only on pushes to `main`)

Pull requests only build; they do not deploy.

## Still failing?

| Error / symptom | Fix |
|-----------------|-----|
| `Not Found` / `Get Pages site failed` | Source is still “Deploy from a branch” or Pages is disabled → set **Source: GitHub Actions** (step 1). |
| Environment `github-pages` missing | Appears after step 1; re-run workflow. |
| Old site content | Hard-refresh browser; CDN can lag a few minutes. |

Local preview: `pip install -e ".[docs]"` then `mkdocs serve`.
