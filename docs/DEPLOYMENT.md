# Deployment bundle

Hand off Phase A (Σ̂) and Phase B settings to another environment or pipeline.

```python
from pmh import PMHConfig, export_deployment, load_deployment_bundle

bundle = export_deployment(
    trainer.artifact_,
    "deploy/my_site_b",
    pmh_config=PMHConfig.balanced(),
    hook="backbone.layer3",
    nuisance="domain_shift",
    notes="Trained 2026-05-19; target hospital B camera",
)
print(bundle.manifest)

artifact, manifest, pmh_cfg = load_deployment_bundle("deploy/my_site_b")
```

## Bundle layout

| File | Contents |
|------|----------|
| `sigma_task.pt` / `.json` | Estimated Σ̂ + preflight |
| `manifest.json` | `matching_pmh_version`, method, dim, hook, nuisance |
| `pmh_config.json` | Optional `PMHConfig` (weight, cap, warmup) |
| `README.txt` | Operator quick reference |

## Load and train

```python
from pmh import PMHTrainer, PMHConfig

artifact, manifest, pmh_cfg = load_deployment_bundle("deploy/my_site_b")
trainer = PMHTrainer.from_artifact(
    model,
    artifact,
    hook=manifest.get("hook") or "backbone",
    pmh_config=pmh_cfg or PMHConfig.balanced(),
)
trainer.fit(train_loader, epochs=20)
```

Mode B (sklearn): load `sigma_task.pt` and use `PMHMatcher` / projection — see [G2](GOLDEN_PATHS.md#g2--frozen-features--sklearn).

---

See also [CUSTOM_GEOMETRY.md](CUSTOM_GEOMETRY.md) · [CLI](cli.md)
