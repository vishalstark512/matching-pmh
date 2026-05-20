"""G3 golden path — replace texts and model with yours."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pmh import check_applicability, robust_fit_text_domains


class ToyTokenizer:
    def __call__(self, texts, **kw):
        import hashlib
        rows = []
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            torch.manual_seed(h % (2**31))
            rows.append(torch.randn(32))
        return {"input_ids": torch.stack(rows)}


class ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.lm_head = nn.Linear(32, 3)

    def forward(self, input_ids=None, labels=None, output_hidden_states=False, **kw):
        h = self.body(input_ids)
        return type("O", (), {"logits": self.lm_head(h), "hidden_states": (h,)})


class DS(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = ToyTokenizer()([self.texts[i]])["input_ids"][0]
        return {"input_ids": ids, "labels": torch.tensor(self.labels[i])}


def main() -> None:
    texts_a = [f"site A sample {i}" for i in range(60)]
    texts_b = [f"site B sample {i}" for i in range(60)]
    labels = [i % 3 for i in range(60)]
    loader = DataLoader(DS(texts_a, labels), batch_size=8, shuffle=True)

    print(check_applicability(stack="hf", n_source=60, n_target=60).summary())

    out = robust_fit_text_domains(
        ToyLM(), ToyTokenizer(), loader,
        source_texts=texts_a, target_texts=texts_b, epochs=2, rank=4,
    )
    print(out.stats)
    print(out.preflight_message)


if __name__ == "__main__":
    main()
