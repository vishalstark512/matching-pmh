# PMHTrainer

Phase **A** (estimate \(\hat\Sigma_{\mathrm{task}}\)) + Phase **B** (train with `PMHLoss`). For frozen features use [PMHMatcher](index.md#pmhmatcher-sklearn) instead.

Paper blocks: [Recipe cards](../recipes/README.md) · [PAPER_ALIGNMENT](../PAPER_ALIGNMENT.md).

::: pmh.trainer.PMHTrainer
    options:
      members:
        - __init__
        - estimate
        - fit
        - training_step
        - measure_trajectory_tdi
        - add_artifact
        - artifact_
        - callback
        - method
        - nuisance
        - encoder
        - head
        - pmh_loss_
      show_root_heading: true
      show_source: false

## Related

- `PMHConfig` — see [training.md](../training.md)
- `compare_arms` — falsification training runs ([Walkthrough 8](../walkthroughs/08-falsification-controls.md))
- `HFPMHTrainer` — [integrations-hf-trainer.md](../integrations-hf-trainer.md)
