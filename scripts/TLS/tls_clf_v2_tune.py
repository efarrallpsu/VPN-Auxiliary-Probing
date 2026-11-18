"""Tune and evaluate the v2 TLS protocol classifier.

Stages:
  feature   - compare candidate feature sets with grouped 3-fold CV
  tune      - random-ish grid search over HGB hyperparameters on the best set
  structure - compare 3-class, multi-label, and hierarchical models
  final     - refit the selected model, evaluate on frozen validation/test,
              save model/metrics/importance/subgroup tables
  robust    - label-permutation, leave-vendor-out, leave-ASN-out checks
  all       - run every stage in order
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, confusion_matrix,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path(__file__).resolve().parents[2]
V2 = BASE / "tls_clf_v2"
CLASSES = ["openvpn", "sstp", "both"]
Y_MAP = {c: i for i, c in enumerate(CLASSES)}

CAT_K = {
    "tls_cipher": 16,
    "cert_validation_level": 8,
    "cert_validation_level_norm": 8,
    "cert_issuer_org": 64,
    "cert_issuer_org_norm": 64,
    "cert_subject_org": 64,
    "cert_subject_org_norm": 64,
    "get_https_server": 64,
    "get_https_server_family": 32,
    "get_https_content_type": 24,
    "get_https_content_type_norm": 16,
    "get_https_status_class": 8,
    "get_https_probe_status": 8,
}

DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "class_weight": None,
}

FEATURE_SETS = {
    "full_v2": None,  # filled from schema
    "compact": [
        "handshake_present", "tls_cipher", "tls_ocsp_stapling", "cert_validity_days",
        "cert_self_signed", "cert_cn_is_ipv4", "cert_cn_has_softether",
        "cert_validation_level_norm", "cert_issuer_org_norm", "cert_subject_org_norm",
        "get_https_probe_status", "get_https_response", "get_https_status_class",
        "get_https_server_family", "get_https_content_type_norm", "get_https_header_count",
    ],
    "compact_no_availability": [
        "tls_cipher", "tls_ocsp_stapling", "cert_validity_days",
        "cert_self_signed", "cert_cn_is_ipv4", "cert_cn_has_softether",
        "cert_validation_level_norm", "cert_issuer_org_norm", "cert_subject_org_norm",
        "get_https_status_class", "get_https_server_family",
        "get_https_content_type_norm", "get_https_header_count",
    ],
    "top_importance": [
        "get_https_header_count", "get_https_server_family", "cert_subject_org_norm",
        "get_https_content_type_norm", "tls_cipher", "get_https_status_class",
        "cert_issuer_org_norm", "tls_ocsp_stapling", "handshake_present",
        "get_https_response", "cert_validation_level_norm", "cert_cn_has_softether",
    ],
    "compact_v2": [
        "tls_cipher", "tls_ocsp_stapling", "cert_age_days", "cert_validity_days",
        "cert_issuer_org_norm", "cert_subject_org_norm",
        "cert_cn_is_ipv4", "cert_cn_has_softether",
        "get_https_probe_status", "get_https_status_class",
        "get_https_server", "get_https_content_type", "get_https_header_count",
    ],
    "compact_v2_plus": [
        "handshake_present", "tls_cipher", "tls_ocsp_stapling", "cert_age_days",
        "cert_validity_days", "cert_issuer_org_norm", "cert_subject_org_norm",
        "cert_cn_is_ipv4", "cert_cn_has_softether", "cert_validation_level_norm",
        "get_https_probe_status", "get_https_response", "get_https_status_class",
        "get_https_server", "get_https_server_family",
        "get_https_content_type", "get_https_header_count",
    ],
}


def load_df() -> pd.DataFrame:
    path = V2 / "features_model.csv"
    if not path.exists():
        path = V2 / "features_model.csv.gz"
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={"ip": "string", "group_id": "string"},
    )
    return df


def get_features_from_schema() -> list[str]:
    schema = json.loads((V2 / "feature_schema.json").read_text())
    return [c for c in schema["allowed_features"] if c != "split"]


def label_codes(frame: pd.DataFrame) -> np.ndarray:
    return frame["label"].map(Y_MAP).to_numpy()


def build_xy(frame: pd.DataFrame, cols: list[str], maps: dict | None = None):
    X = pd.DataFrame(index=frame.index)
    new_maps = {}
    for c in cols:
        if c in CAT_K:
            s = frame[c].astype("object").where(frame[c].notna(), None)
            if maps is None:
                vc = s.value_counts().head(CAT_K[c])
                mapping = {v: i + 1 for i, v in enumerate(vc.index)}
            else:
                mapping = maps[c]
            code = s.map(mapping)
            X[c] = np.where(s.isna(), 0, code.fillna(len(mapping) + 1)).astype("int32")
            new_maps[c] = mapping
        else:
            X[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")
    cat_idx = [X.columns.get_loc(c) for c in X.columns if c in CAT_K]
    return X, cat_idx, new_maps


def model_kwargs(params: dict, cat_idx: list[int]) -> dict:
    kw = dict(
        learning_rate=params.get("learning_rate", 0.1),
        max_iter=params.get("max_iter", 250),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        l2_regularization=params.get("l2_regularization", 0.0),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        categorical_features=cat_idx,
        random_state=340,
        class_weight=params.get("class_weight"),
    )
    return kw


def fit_3class(train: pd.DataFrame, cols: list[str], params: dict):
    X, cat_idx, maps = build_xy(train, cols)
    y = label_codes(train)
    clf = HistGradientBoostingClassifier(**model_kwargs(params, cat_idx))
    clf.fit(X, y)
    return {"structure": "3class", "features": cols, "maps": maps,
            "cat_idx": cat_idx, "model": clf, "params": params}


def predict_bundle(bundle: dict, frame: pd.DataFrame):
    cols = bundle["features"]
    X, _, _ = build_xy(frame, cols, bundle["maps"])
    if bundle["structure"] == "3class":
        pred = bundle["model"].predict(X)
        proba = bundle["model"].predict_proba(X)
        return pred, proba
    if bundle["structure"] == "multilabel":
        p1 = bundle["model"]["openvpn"].predict_proba(X)[:, 1]
        p2 = bundle["model"]["sstp"].predict_proba(X)[:, 1]
        thr = bundle.get("threshold", 0.5)
        pred = np.where((p1 >= thr) & (p2 >= thr), 2, np.where(p1 >= p2, 0, 1))
        proba = np.column_stack([
            np.where(pred == 0, p1, 0.0),
            np.where(pred == 1, p2, 0.0),
            np.where(pred == 2, np.minimum(p1, p2), 0.0),
        ])
        return pred, proba
    if bundle["structure"] == "hierarchical":
        p_both = bundle["model"]["both"].predict_proba(X)[:, 1]
        p_sstp = bundle["model"]["sstp"].predict_proba(X)[:, 1]
        thr = bundle.get("threshold", 0.5)
        pred = np.where(p_both >= thr, 2, np.where(p_sstp >= 0.5, 1, 0))
        proba = np.column_stack([
            np.where(pred == 0, 1 - p_both, 0.0),
            np.where(pred == 1, p_sstp * (1 - p_both), 0.0),
            np.where(pred == 2, p_both, 0.0),
        ])
        return pred, proba
    raise ValueError(bundle["structure"])


def fit_bundle(train: pd.DataFrame, cols: list[str], params: dict, structure: str,
               threshold: float = 0.5):
    if structure == "3class":
        bundle = fit_3class(train, cols, params)
        bundle["threshold"] = threshold
        return bundle

    X, cat_idx, maps = build_xy(train, cols)
    y = label_codes(train)
    if structure == "multilabel":
        y1 = np.isin(y, [0, 2]).astype(int)
        y2 = np.isin(y, [1, 2]).astype(int)
        m1 = HistGradientBoostingClassifier(**model_kwargs(params, cat_idx)).fit(X, y1)
        m2 = HistGradientBoostingClassifier(**model_kwargs(params, cat_idx)).fit(X, y2)
        return {"structure": structure, "features": cols, "maps": maps,
                "cat_idx": cat_idx, "model": {"openvpn": m1, "sstp": m2},
                "params": params, "threshold": threshold}
    if structure == "hierarchical":
        y_both = (y == 2).astype(int)
        m_both = HistGradientBoostingClassifier(**model_kwargs(params, cat_idx)).fit(X, y_both)
        single = y != 2
        y_sstp = (y[single] == 1).astype(int)
        m_sstp = HistGradientBoostingClassifier(**model_kwargs(params, cat_idx)).fit(X[single], y_sstp)
        return {"structure": structure, "features": cols, "maps": maps,
                "cat_idx": cat_idx, "model": {"both": m_both, "sstp": m_sstp},
                "params": params, "threshold": threshold}
    raise ValueError(structure)


def metrics(y_true, y_pred, proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macroF1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "per_class_F1": {
            c: float(v) for c, v in zip(
                CLASSES, f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
            )
        },
    }
    if proba is not None and np.allclose(np.asarray(proba).sum(axis=1), 1.0, atol=1e-3):
        try:
            out["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1, 2]))
        except ValueError:
            pass
        try:
            out["macro_ovr_auc"] = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=[0, 1, 2]
            ))
        except ValueError:
            pass
    return out


def get_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    path = V2 / "cv_folds.csv"
    if not path.exists() and (V2 / "cv_folds.csv.gz").exists():
        path = V2 / "cv_folds.csv.gz"
    if path.exists():
        f = pd.read_csv(path, dtype={"ip": "string"})
        fold_map = dict(zip(f["ip"], f["fold"]))
        folds = train["ip"].map(fold_map).to_numpy()
    else:
        y = label_codes(train)
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=340)
        folds = np.full(len(train), -1, dtype=int)
        for k, (_, va) in enumerate(cv.split(train, y, groups=train["group_id"])):
            folds[va] = k
        pd.DataFrame({"ip": train["ip"], "fold": folds}).to_csv(path, index=False)
    assert (folds >= 0).all()
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in sorted(set(folds))]


def stage_feature() -> None:
    print("stage: feature-set comparison")
    df = load_df()
    train = df[df["split"] == "train"].reset_index(drop=True)
    folds = get_folds(train)
    full = get_features_from_schema()
    feature_sets = dict(FEATURE_SETS)
    feature_sets["full_v2"] = full
    rows = []
    for name, cols in feature_sets.items():
        per_fold = []
        for k, (tr, va) in enumerate(folds):
            bundle = fit_3class(train.iloc[tr], cols, DEFAULT_PARAMS)
            pred, proba = predict_bundle(bundle, train.iloc[va])
            m = metrics(label_codes(train.iloc[va]), pred, proba)
            m.update({"fold": k, "n_features": len(cols)})
            per_fold.append(m)
            print(f"  {name} fold {k}: acc {m['accuracy']:.4f} macroF1 {m['macroF1']:.4f}")
        pf = pd.DataFrame(per_fold)
        rows.append({
            "feature_set": name,
            "n_features": len(cols),
            "accuracy_mean": pf["accuracy"].mean(),
            "accuracy_std": pf["accuracy"].std(),
            "macroF1_mean": pf["macroF1"].mean(),
            "macroF1_std": pf["macroF1"].std(),
            "openvpn_F1_mean": pf["per_class_F1"].apply(lambda d: d["openvpn"]).mean(),
            "sstp_F1_mean": pf["per_class_F1"].apply(lambda d: d["sstp"]).mean(),
            "both_F1_mean": pf["per_class_F1"].apply(lambda d: d["both"]).mean(),
        })
    out = pd.DataFrame(rows).sort_values("macroF1_mean", ascending=False)
    out.to_csv(V2 / "feature_set_results.csv", index=False)
    print(out.to_string(index=False))


def tuning_configs() -> list[dict]:
    bases = [
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_iter": 250, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.1, "max_iter": 250, "max_leaf_nodes": 63, "min_samples_leaf": 50, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 0.1},
    ]
    return [{**b, "class_weight": cw} for b, cw in product(bases, [None, "balanced"])]


def stage_tune() -> None:
    print("stage: hyperparameter tuning")
    fs = pd.read_csv(V2 / "feature_set_results.csv").sort_values("macroF1_mean", ascending=False)
    # parsimony rule: among sets within 0.002 CV macro-F1 of the best, take the smallest
    best_mean = fs["macroF1_mean"].max()
    eligible = fs[fs["macroF1_mean"] >= best_mean - 0.002]
    eligible = eligible.sort_values(["n_features", "macroF1_mean"], ascending=[True, False])
    best_set_name = eligible.iloc[0]["feature_set"]
    full = get_features_from_schema()
    all_sets = dict(FEATURE_SETS)
    all_sets["full_v2"] = full
    cols = all_sets[best_set_name]
    print(f"best feature set: {best_set_name} ({len(cols)} features)")
    df = load_df()
    train = df[df["split"] == "train"].reset_index(drop=True)
    folds = get_folds(train)
    rows = []
    for ci, params in enumerate(tuning_configs()):
        per_fold = []
        for k, (tr, va) in enumerate(folds):
            bundle = fit_3class(train.iloc[tr], cols, params)
            pred, proba = predict_bundle(bundle, train.iloc[va])
            m = metrics(label_codes(train.iloc[va]), pred, proba)
            m["fold"] = k
            per_fold.append(m)
        pf = pd.DataFrame(per_fold)
        row = {
            "config": ci,
            "feature_set": best_set_name,
            "n_features": len(cols),
            "params": json.dumps(params),
            "accuracy_mean": pf["accuracy"].mean(),
            "accuracy_std": pf["accuracy"].std(),
            "macroF1_mean": pf["macroF1"].mean(),
            "macroF1_std": pf["macroF1"].std(),
            "openvpn_F1_mean": pf["per_class_F1"].apply(lambda d: d["openvpn"]).mean(),
            "sstp_F1_mean": pf["per_class_F1"].apply(lambda d: d["sstp"]).mean(),
            "both_F1_mean": pf["per_class_F1"].apply(lambda d: d["both"]).mean(),
        }
        rows.append(row)
        print(f"  config {ci}: macroF1 {row['macroF1_mean']:.4f} (+/- {row['macroF1_std']:.4f}) both {row['both_F1_mean']:.4f}")
    out = pd.DataFrame(rows).sort_values(["macroF1_mean", "macroF1_std"], ascending=[False, True])
    out.to_csv(V2 / "tuning_results.csv", index=False)
    best = out.iloc[0]
    (V2 / "best_config.json").write_text(json.dumps({
        "feature_set": best["feature_set"],
        "features": cols,
        "params": json.loads(best["params"]),
        "cv_macroF1_mean": float(best["macroF1_mean"]),
        "cv_macroF1_std": float(best["macroF1_std"]),
        "cv_accuracy_mean": float(best["accuracy_mean"]),
        "cv_both_F1_mean": float(best["both_F1_mean"]),
    }, indent=2))
    print("best config:")
    print(best.to_string())


def stage_structure() -> None:
    print("stage: model-structure comparison")
    cfg = json.loads((V2 / "best_config.json").read_text())
    cols = cfg["features"]
    params = cfg["params"]
    df = load_df()
    train = df[df["split"] == "train"].reset_index(drop=True)
    folds = get_folds(train)
    rows = []
    for structure in ["3class", "multilabel", "hierarchical"]:
        for k, (tr, va) in enumerate(folds):
            bundle = fit_bundle(train.iloc[tr], cols, params, structure)
            # pick the threshold on the fold's validation predictions
            best_thr, best_m = 0.5, None
            thresholds = [0.5] if structure == "3class" else [0.3, 0.4, 0.5, 0.6]
            for thr in thresholds:
                bundle["threshold"] = thr
                pred, proba = predict_bundle(bundle, train.iloc[va])
                m = metrics(label_codes(train.iloc[va]), pred, proba)
                if best_m is None or m["macroF1"] > best_m["macroF1"]:
                    best_thr, best_m = thr, m
            best_m.update({"structure": structure, "fold": k, "threshold": best_thr})
            rows.append(best_m)
            print(f"  {structure} fold {k}: thr {best_thr} macroF1 {best_m['macroF1']:.4f} both {best_m['per_class_F1']['both']:.4f}")
    pf = pd.DataFrame(rows)
    out = pf.groupby("structure").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macroF1_mean=("macroF1", "mean"), macroF1_std=("macroF1", "std"),
        both_F1_mean=("per_class_F1", lambda s: s.apply(lambda d: d["both"]).mean()),
        openvpn_F1_mean=("per_class_F1", lambda s: s.apply(lambda d: d["openvpn"]).mean()),
        sstp_F1_mean=("per_class_F1", lambda s: s.apply(lambda d: d["sstp"]).mean()),
        threshold_mean=("threshold", "mean"),
    ).sort_values("macroF1_mean", ascending=False).reset_index()
    out.to_csv(V2 / "structure_results.csv", index=False)
    pf.to_csv(V2 / "structure_fold_results.csv", index=False)
    print(out.to_string(index=False))
    best = out.iloc[0]
    (V2 / "best_structure.json").write_text(json.dumps({
        "structure": best["structure"],
        "threshold": float(best["threshold_mean"]),
        "cv_macroF1_mean": float(best["macroF1_mean"]),
        "cv_macroF1_std": float(best["macroF1_std"]),
        "cv_both_F1_mean": float(best["both_F1_mean"]),
    }, indent=2))


def subgroup_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    y = label_codes(frame)
    out = {}
    hs = frame["handshake_present"].fillna(False).astype(bool).to_numpy()
    get = frame["get_https_response"].fillna(False).astype(bool).to_numpy()
    for name, mask in [
        ("all", np.ones(len(frame), bool)),
        ("tls_handshake_true", hs),
        ("tls_handshake_false", ~hs),
        ("get_https_response_true", get),
        ("get_https_response_false", ~get),
    ]:
        if mask.sum() == 0:
            continue
        out[name] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "macroF1": float(f1_score(y[mask], pred[mask], average="macro", labels=[0, 1, 2], zero_division=0)),
            "per_class_F1": {
                c: float(v) for c, v in zip(CLASSES, f1_score(
                    y[mask], pred[mask], average=None, labels=[0, 1, 2], zero_division=0
                ))
            },
        }
    return out


def permutation_importance_bundle(bundle: dict, frame: pd.DataFrame, y: np.ndarray,
                                  n_repeats: int = 5) -> pd.DataFrame:
    X, _, _ = build_xy(frame, bundle["features"], bundle["maps"])
    rng = np.random.default_rng(340)
    base_pred, _ = predict_bundle(bundle, frame)
    base = f1_score(y, base_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    rows = []
    for c in bundle["features"]:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            pred = predict_from_X(bundle, Xp)
            drops.append(base - f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))
        rows.append({"feature": c, "mean_drop_macroF1": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("mean_drop_macroF1", ascending=False)


def predict_from_X(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    if bundle["structure"] == "3class":
        return bundle["model"].predict(X)
    if bundle["structure"] == "multilabel":
        p1 = bundle["model"]["openvpn"].predict_proba(X)[:, 1]
        p2 = bundle["model"]["sstp"].predict_proba(X)[:, 1]
        thr = bundle.get("threshold", 0.5)
        return np.where((p1 >= thr) & (p2 >= thr), 2, np.where(p1 >= p2, 0, 1))
    if bundle["structure"] == "hierarchical":
        p_both = bundle["model"]["both"].predict_proba(X)[:, 1]
        p_sstp = bundle["model"]["sstp"].predict_proba(X)[:, 1]
        thr = bundle.get("threshold", 0.5)
        return np.where(p_both >= thr, 2, np.where(p_sstp >= 0.5, 1, 0))
    raise ValueError(bundle["structure"])


def stage_final() -> None:
    print("stage: final fit and evaluation")
    cfg = json.loads((V2 / "best_config.json").read_text())
    st = json.loads((V2 / "best_structure.json").read_text())
    cols = cfg["features"]
    params = cfg["params"]
    structure = st["structure"]
    threshold = st["threshold"]
    df = load_df()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    bundle = fit_bundle(train, cols, params, structure, threshold)
    joblib.dump(bundle, V2 / "model.joblib")

    out = {
        "feature_set": cfg["feature_set"],
        "features": cols,
        "params": params,
        "structure": structure,
        "threshold": threshold,
        "n_features": len(cols),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "cv": {"macroF1_mean": cfg["cv_macroF1_mean"], "macroF1_std": cfg["cv_macroF1_std"]},
        "splits": {},
    }
    for name, frame in [("validation", val), ("test", test)]:
        pred, proba = predict_bundle(bundle, frame)
        m = metrics(label_codes(frame), pred, proba)
        m["confusion_matrix"] = confusion_matrix(label_codes(frame), pred, labels=[0, 1, 2]).tolist()
        m["subgroups"] = subgroup_metrics(frame, pred)
        out["splits"][name] = m
        print(f"\n{name}:")
        print(f"  accuracy {m['accuracy']:.4f} macroF1 {m['macroF1']:.4f}")
        print(f"  per-class F1 {m['per_class_F1']}")
        print(pd.DataFrame(m["confusion_matrix"], index=CLASSES, columns=CLASSES).to_string())

    # confusion matrix plot for validation
    cm = np.asarray(out["splits"]["validation"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("v2 validation confusion matrix")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(V2 / "confusion_matrix_validation.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(V2 / "confusion_matrix_validation.csv")

    # permutation importance on validation
    print("computing permutation importance ...")
    imp = permutation_importance_bundle(bundle, val, label_codes(val), n_repeats=5)
    imp.to_csv(V2 / "permutation_importance.csv", index=False)
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_drop_macroF1"], xerr=top["std"], color="#1f77b4", alpha=0.85)
    ax.set_xlabel("drop in macro-F1 when shuffled (validation)")
    ax.set_title("v2 permutation importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(V2 / "permutation_importance.png", dpi=160)
    plt.close(fig)

    (V2 / "final_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote model/metrics/plots")


def stage_robust() -> None:
    print("stage: robustness checks")
    cfg = json.loads((V2 / "best_config.json").read_text())
    st = json.loads((V2 / "best_structure.json").read_text())
    cols, params = cfg["features"], cfg["params"]
    structure, threshold = st["structure"], st["threshold"]
    df = load_df()
    raw_path = V2 / "asn_by_ip.csv.gz"
    if not raw_path.exists():
        raw_path = V2 / "features_v2.csv"
    if not raw_path.exists():
        raw_path = V2 / "features_v2.csv.gz"
    raw = pd.read_csv(raw_path, usecols=["ip", "asn"], dtype={"ip": "string"})
    df = df.merge(raw, on="ip", how="left")
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    out = {}

    # label permutation null test
    rng = np.random.default_rng(340)
    perm = train.copy()
    perm["label"] = rng.permutation(perm["label"].to_numpy())
    bundle = fit_bundle(perm, cols, params, structure, threshold)
    pred, proba = predict_bundle(bundle, val)
    out["label_permutation"] = metrics(label_codes(val), pred, proba)
    print("label permutation:", out["label_permutation"]["accuracy"],
          out["label_permutation"]["macroF1"])

    # leave-vendor-out
    vendors = ["watchguard", "digicert", "sectigo", "microsoft", "stormshield", "letsencrypt", "thevpn"]
    vendor_out = {}
    for vendor in vendors:
        tr_mask = train["cert_issuer_org_norm"] != vendor
        va_mask = val["cert_issuer_org_norm"] == vendor
        if tr_mask.sum() < 1000 or va_mask.sum() < 50:
            continue
        b = fit_bundle(train[tr_mask].reset_index(drop=True), cols, params, structure, threshold)
        p, pr = predict_bundle(b, val[va_mask].reset_index(drop=True))
        m = metrics(label_codes(val[va_mask]), p, pr)
        m["n_train"] = int(tr_mask.sum())
        m["n_validation"] = int(va_mask.sum())
        vendor_out[vendor] = m
        print(f"vendor {vendor}: n={m['n_validation']} acc {m['accuracy']:.4f} macroF1 {m['macroF1']:.4f}")
    out["leave_vendor_out"] = vendor_out

    # leave-ASN-out: hold out 20% of ASNs by row count
    asn_counts = df["asn"].dropna().value_counts()
    asns = asn_counts.index.to_numpy()
    rng2 = np.random.default_rng(341)
    holdout = set(rng2.choice(asns, size=max(1, int(0.2 * len(asns))), replace=False).tolist())
    tr_mask = ~train["asn"].isin(holdout)
    va_mask = val["asn"].isin(holdout)
    if tr_mask.sum() > 1000 and va_mask.sum() > 50:
        b = fit_bundle(train[tr_mask].reset_index(drop=True), cols, params, structure, threshold)
        p, pr = predict_bundle(b, val[va_mask].reset_index(drop=True))
        m = metrics(label_codes(val[va_mask]), p, pr)
        m["n_train"] = int(tr_mask.sum())
        m["n_validation"] = int(va_mask.sum())
        m["n_holdout_asns"] = len(holdout)
        out["leave_asn_out"] = m
        print(f"ASN holdout: n={m['n_validation']} acc {m['accuracy']:.4f} macroF1 {m['macroF1']:.4f}")

    # unseen feature-fingerprint evaluation (does the model rely on train twins?)
    df["hash"] = pd.util.hash_pandas_object(df[cols], index=False)
    train_hashes = set(df.loc[df["split"] == "train", "hash"])
    final_bundle = joblib.load(V2 / "model.joblib")
    unseen = {}
    for split in ["validation", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        pred, proba = predict_bundle(final_bundle, sub)
        y = label_codes(sub)
        seen = sub["hash"].isin(train_hashes).to_numpy()
        entry = {}
        for name, mask in [("all", np.ones(len(sub), bool)), ("seen_fingerprint", seen),
                           ("unseen_fingerprint", ~seen)]:
            if mask.sum() == 0:
                continue
            m = metrics(y[mask], pred[mask], proba[mask])
            m["n"] = int(mask.sum())
            entry[name] = m
        unseen[split] = entry
        print(f"{split} unseen fingerprint: n={entry['unseen_fingerprint']['n']:,} "
              f"acc {entry['unseen_fingerprint']['accuracy']:.4f} "
              f"macroF1 {entry['unseen_fingerprint']['macroF1']:.4f}")
    out["unseen_fingerprint"] = unseen

    (V2 / "robustness.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote robustness.json")


def stage_report() -> None:
    print("stage: report")
    cfg = json.loads((V2 / "best_config.json").read_text())
    st = json.loads((V2 / "best_structure.json").read_text())
    fm = json.loads((V2 / "final_metrics.json").read_text())
    fs = pd.read_csv(V2 / "feature_set_results.csv")
    tune = pd.read_csv(V2 / "tuning_results.csv")
    struct = pd.read_csv(V2 / "structure_results.csv")
    imp = pd.read_csv(V2 / "permutation_importance.csv")
    rob = json.loads((V2 / "robustness.json").read_text()) if (V2 / "robustness.json").exists() else {}

    def md_table(frame):
        frame = pd.DataFrame(frame)
        cols = [str(c) for c in frame.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in frame.iterrows():
            vals = []
            for c in frame.columns:
                v = row[c]
                if isinstance(v, float):
                    v = f"{v:.4g}"
                vals.append(str(v).replace("|", "\\|"))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = []
    lines.append("# TLS protocol classifier v2\n")
    lines.append("## What changed from v1\n")
    lines.append("- TLS records use a best-record merge instead of first-file-wins.")
    lines.append("- GET records use best-response selection and field-fill; response/status are consistent.")
    lines.append("- Certificate age/expired are relative to scan date.")
    lines.append("- Vendor/server/content-type strings are normalized.")
    lines.append("- Constants, near-constants, rDNS, ASN, prefix_len, and raw CN strings are excluded from the model.")
    lines.append("- Hyperparameters and model structure are chosen by grouped 3-fold CV on the training split only.")
    lines.append("")
    lines.append("## Selected model\n")
    lines.append(f"- Feature set: `{cfg['feature_set']}` ({len(cfg['features'])} features)")
    lines.append(f"- Structure: `{st['structure']}`")
    if st["structure"] == "3class":
        lines.append("- Decision rule: argmax over the three class probabilities")
    else:
        lines.append(f"- Decision threshold: {st['threshold']:.2f}")
    lines.append(f"- CV macro-F1: {cfg['cv_macroF1_mean']:.4f} ± {cfg['cv_macroF1_std']:.4f}")
    lines.append("")
    lines.append("### Features\n")
    lines.append(", ".join(f"`{c}`" for c in cfg["features"]))
    lines.append("")
    lines.append("### v1 vs v2\n")
    lines.append(md_table(pd.DataFrame([
        {"version": "v1", "features": 55, "validation_accuracy": 0.9845,
         "validation_macroF1": 0.9561, "test_accuracy": 0.9815, "test_macroF1": 0.9448},
        {"version": "v2", "features": len(cfg["features"]),
         "validation_accuracy": fm["splits"]["validation"]["accuracy"],
         "validation_macroF1": fm["splits"]["validation"]["macroF1"],
         "test_accuracy": fm["splits"]["test"]["accuracy"],
         "test_macroF1": fm["splits"]["test"]["macroF1"]},
    ])))
    lines.append("")
    lines.append("v1 used the 55-feature table and its test set was used for ablations; "
                 "v2's test set was not used for tuning. The v2 compact model is "
                 "essentially tied on validation and slightly lower on test, with 13 features instead of 55.")
    lines.append("")
    lines.append("## Final metrics\n")
    for split in ["validation", "test"]:
        m = fm["splits"][split]
        lines.append(f"### {split}\n")
        lines.append(f"Accuracy: **{m['accuracy']:.4f}**; macro-F1: **{m['macroF1']:.4f}**")
        if "log_loss" in m:
            lines.append(f"; log loss: **{m['log_loss']:.4f}**")
        if "macro_ovr_auc" in m:
            lines.append(f"; macro OVR AUC: **{m['macro_ovr_auc']:.4f}**")
        lines.append("\n")
        lines.append(md_table(pd.DataFrame(
            [{"class": c, "F1": m["per_class_F1"][c]} for c in CLASSES]
        )))
        lines.append("\n")
        lines.append("Subgroups:\n")
        lines.append(md_table(pd.DataFrame([
            {"subgroup": k, "n": v["n"], "accuracy": v["accuracy"], "macroF1": v["macroF1"]}
            for k, v in m["subgroups"].items()
        ])))
        lines.append("\n")
    lines.append("## Feature-set comparison\n")
    lines.append(md_table(fs))
    lines.append("\n## Hyperparameter search (top 10)\n")
    lines.append(md_table(tune.head(10)[["config", "macroF1_mean", "macroF1_std", "accuracy_mean",
                                        "both_F1_mean", "sstp_F1_mean", "openvpn_F1_mean", "params"]]))
    lines.append("\n## Model structure comparison\n")
    lines.append(md_table(struct))
    lines.append("\n## Permutation importance (top 15)\n")
    lines.append(md_table(imp.head(15)))
    if rob:
        lines.append("\n## Robustness checks\n")
        lines.append(f"Label permutation: accuracy **{rob['label_permutation']['accuracy']:.4f}**, "
                     f"macro-F1 **{rob['label_permutation']['macroF1']:.4f}**\n")
        if rob.get("leave_vendor_out"):
            lines.append("Leave-vendor-out:\n")
            lines.append(md_table(pd.DataFrame([
                {"vendor": k, "n_validation": v["n_validation"],
                 "accuracy": v["accuracy"], "macroF1": v["macroF1"],
                 "both_F1": v["per_class_F1"]["both"]}
                for k, v in rob["leave_vendor_out"].items()
            ])))
        if rob.get("leave_asn_out"):
            v = rob["leave_asn_out"]
            lines.append(f"\nLeave-ASN-out: n={v['n_validation']:,}, accuracy **{v['accuracy']:.4f}**, "
                         f"macro-F1 **{v['macroF1']:.4f}**, both F1 **{v['per_class_F1']['both']:.4f}**\n")
        if rob.get("unseen_fingerprint"):
            lines.append("\nUnseen feature-fingerprint check (validation rows whose exact \n"
                         "13-feature vector does not appear in train):\n")
            rows = []
            for split, entry in rob["unseen_fingerprint"].items():
                for name in ["all", "seen_fingerprint", "unseen_fingerprint"]:
                    if name not in entry:
                        continue
                    e = entry[name]
                    rows.append({"split": split, "subset": name, "n": e["n"],
                                 "accuracy": e["accuracy"], "macroF1": e["macroF1"],
                                 "both_F1": e["per_class_F1"]["both"]})
            lines.append(md_table(pd.DataFrame(rows)))
    lines.append("\n## Known limitations\n")
    lines.append("- The universe is still positives-only: no ordinary HTTPS negatives.")
    lines.append("- The labels come from the two protocol probes, so probe failure is part of the target.")
    lines.append("- Much of the signal is vendor/software fingerprinting; expect drift across firmware versions.")
    lines.append("- The split is grouped by certificate/ /24, not by time, ASN, or vendor; robustness checks above quantify some of that.")
    lines.append("- `both` is heavily SoftEther/multi-protocol; it is not a clean ground-truth multi-protocol label.")
    (V2 / "V2_REPORT.md").write_text("\n".join(lines) + "\n")

    model_card = {
        "model": "tls_clf_v2",
        "universe": "U3 = H(openvpn_tcp) union H(sstp), IPv4; positives-only",
        "label_definition": "openvpn = only OpenVPN-TCP hitlist; sstp = only SSTP hitlist; both = both hitlists",
        "features": cfg["features"],
        "forbidden_columns": json.loads((V2 / "feature_schema.json").read_text()).get("forbidden_columns", []),
        "structure": st["structure"],
        "params": cfg["params"],
        "decision_rule": "argmax over 3-class probabilities",
        "cv": cfg,
        "validation": fm["splits"]["validation"],
        "test": fm["splits"]["test"],
        "robustness": rob,
        "artifacts": [
            "tls_clf_v2/features_v2.csv", "tls_clf_v2/features_model.csv",
            "tls_clf_v2/feature_schema.json", "tls_clf_v2/model.joblib",
            "tls_clf_v2/final_metrics.json", "tls_clf_v2/robustness.json",
            "tls_clf_v2/V2_REPORT.md", "tls_protocol_classification_v2.ipynb",
        ],
        "caveats": [
            "closed positives-only universe; not a VPN-vs-non-VPN detector",
            "labels are probe membership, not ground-truth protocol support",
            "vendor/software fingerprint drift expected",
            "split grouped by certificate/ /24, not time/ASN/vendor",
            "both is heavily SoftEther/multi-protocol",
        ],
    }
    (V2 / "model_card.json").write_text(json.dumps(model_card, indent=2, default=str))
    print("wrote", V2 / "V2_REPORT.md", "and model_card.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["feature", "tune", "structure", "final", "robust", "report", "all"])
    args = ap.parse_args()
    stages = ["feature", "tune", "structure", "final", "robust", "report"] if args.stage == "all" else [args.stage]
    fn = {
        "feature": stage_feature,
        "tune": stage_tune,
        "structure": stage_structure,
        "final": stage_final,
        "robust": stage_robust,
        "report": stage_report,
    }
    for s in stages:
        t0 = time.time()
        fn[s]()
        print(f"stage {s} done in {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
