# Datasets

## Office-31 features

Install: `pip install "matching-pmh[vision]"` (torchvision).

```python
from pmh.datasets.office31 import extract_office31_features, list_office31_domains

x_a, y_a = extract_office31_features("/data/office31", "amazon", max_samples=2000)
x_d, y_d = extract_office31_features("/data/office31", "dslr", max_samples=2000)
```

CLI example:

```bash
python examples/06_office31_sklearn.py --office31-root /data/office31 --source amazon --target dslr
```

Expect layout: `root/amazon/<class>/...` (ImageFolder).
