# VPN Auxiliary Probing

Auxiliary TLS/HTTP/rDNS fingerprinting for VPN protocol classification on port 443.

This is the minimal code-only repository for the v2 work. Generated feature
tables, models, plots, and the report notebook are intentionally not committed;
they are recreated by the scripts below.

## Contents

- `tls_protocol_classification.ipynb` — original V1 notebook.
- `tls_clf_splits/*.csv.gz` — frozen grouped 70/20/10 split tables.
- `scripts/TLS/tls_clf_v2_build.py` — rebuilds the v2 feature table from the raw
  VPN ecosystem scans.
- `scripts/TLS/tls_clf_v2_tune.py` — grouped CV, feature-set comparison,
  hyperparameter search, model-structure comparison, final fit, robustness
  checks, and report generation.

## Requirements

The V2 build script expects the raw PAM 2023 VPN ecosystem dataset layout:

```text
output/
hitlists/
tls_clf_splits/
```

The included `tls_clf_splits/*.csv.gz` files are sufficient for the split
assignment. The V1 notebook expects the split files to be decompressed as
`train_70.csv`, `test_20.csv`, and `validation_10_unseen.csv`.

## Reproduce v2

```bash
python scripts/TLS/tls_clf_v2_build.py
python scripts/TLS/tls_clf_v2_tune.py all
```

The first command streams the raw TLS/GET/rDNS/ASN files and writes:

- `tls_clf_v2/features_v2.csv`
- `tls_clf_v2/features_model.csv`
- `tls_clf_v2/feature_schema.json`
- `tls_clf_v2/build_summary.json`

The second command runs the grouped 3-fold CV, tuning, structure comparison,
final fit, robustness checks, and report generation. It writes:

- `tls_clf_v2/model.joblib`
- `tls_clf_v2/final_metrics.json`
- `tls_clf_v2/robustness.json`
- `tls_clf_v2/model_card.json`
- `tls_clf_v2/V2_REPORT.md`
- tuning/CV tables and plots under `tls_clf_v2/`

## V2 summary

- Universe: `H(openvpn_tcp) ∪ H(sstp)`, IPv4, positives only.
- Labels: `openvpn`, `sstp`, `both` from hitlist membership.
- Selected model: 13-feature 3-class HistGradientBoostingClassifier.
- Grouped 3-fold CV macro-F1: **0.9350 ± 0.0127**.
- Frozen validation: accuracy **0.9849**, macro-F1 **0.9559**.
- Frozen test: accuracy **0.9798**, macro-F1 **0.9377**.

The model is a closed-universe attribution model, not a VPN-vs-non-VPN detector.
The v2 report and model card contain the full results and caveats.

## Caveats

- The universe is positives-only: no ordinary HTTPS negatives.
- Labels come from the protocol probes, so probe failure is part of the target.
- Much of the signal is vendor/software fingerprinting; expect drift across
  firmware and web-UI changes.
- The split is grouped by certificate/ /24, not by time, ASN, or vendor.
- `both` is heavily SoftEther/multi-protocol and is not a clean ground-truth
  “supports both protocols” label.
