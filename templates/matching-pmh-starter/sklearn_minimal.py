"""G2 golden path — replace arrays with your embeddings."""

from pmh import check_applicability, evaluate_baseline_vs_pmh, load_g2_demo_arrays

# Office-31-style demo (no download). Swap for your x_source, y_source, x_target, y_target.
x_source, y_source, x_target, y_target = load_g2_demo_arrays(n=500, seed=0)

print(check_applicability(
    stack="sklearn",
    n_source=len(x_source),
    n_target=len(x_target),
).summary())

# Tunable: rank=16, nuisance="domain_shift" (or subspace), compare_to=("coral",)
report = evaluate_baseline_vs_pmh(
    x_source=x_source,
    y_source=y_source,
    x_target=x_target,
    y_target=y_target,
    rank=16,
    nuisance="domain_shift",
    compare_to=("coral",),
)
print(report.summary())
