"""Framework integrations (PyTorch training loops, HF, Lightning)."""

from pmh.integrations.torch import PMHCallback, PMHStepResult, train_epoch_with_pmh

__all__ = [
    "PMHCallback",
    "PMHStepResult",
    "train_epoch_with_pmh",
]


def __getattr__(name: str):
    if name == "PMHLightningCallback":
        from pmh.integrations.lightning import PMHLightningCallback

        return PMHLightningCallback
    if name == "add_pmh_to_loss":
        from pmh.integrations.lightning import add_pmh_to_loss

        return add_pmh_to_loss
    if name in (
        "load_style_pairs_jsonl",
        "encode_style_deltas",
        "encode_texts",
        "estimate_style_sigma",
        "StylePairRecord",
    ):
        from pmh import integrations

        return getattr(integrations.huggingface, name)
    raise AttributeError(name)
