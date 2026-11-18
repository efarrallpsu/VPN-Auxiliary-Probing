"""Build the v2 TLS classifier feature table.

Differences from the original notebook:
  * TLS records are merged with a best-record rule instead of first-file-wins.
  * GET records are chosen by best response and then field-filled from the other scan.
  * Certificate age/expired are relative to the scan date, not a fixed reference date.
  * Common organization/server/content-type strings are normalized.
  * Label/group/id/source columns are kept for audit but marked forbidden for modeling.

The row universe and frozen split assignment are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "output"
HIT = BASE / "hitlists"
SPLIT_DIR = BASE / "tls_clf_splits"
V2 = BASE / "tls_clf_v2"
V2.mkdir(exist_ok=True)

SCAN_DATES = {
    "openvpn": datetime(2022, 8, 1, tzinfo=timezone.utc),
    "sstp": datetime(2022, 8, 9, tzinfo=timezone.utc),
}

TLS_KEYS = [
    "handshake_present", "tls_version", "tls_cipher", "tls_secure_renegotiation", "tls_ticket",
    "tls_ocsp_stapling", "tls_heartbeat_ext", "tls_extended_master_secret", "cke_rsa_present",
    "cke_rsa_params_len", "cert_present", "cert_validity_days", "cert_age_days", "cert_expired",
    "cert_self_signed", "cert_sig_valid", "cert_sig_algorithm", "cert_key_algorithm", "cert_key_bits",
    "cert_key_curve", "cert_san_dns_count", "cert_names_count", "cert_has_wildcard", "cert_cn_is_ipv4",
    "cert_cn_has_softether", "cert_cn_localhost", "cert_subject_eq_issuer", "cert_is_ca",
    "cert_key_encipherment", "cert_digital_signature", "cert_server_auth", "cert_validation_level",
    "cert_subject_cn", "cert_subject_org", "cert_issuer_cn", "cert_issuer_org",
]

HTTPS_KEYS = [
    "get_https_probe_status", "get_https_response", "get_https_status_code",
    "get_https_status_class", "get_https_server", "get_https_content_type",
    "get_https_body_len", "get_https_header_count",
]
HTTP_KEYS = ["get_http_probe_status", "get_http_response", "get_http_status_code", "get_http_server"]
RDNS_KEYS = [
    "rdns_present", "rdns_domain_count", "rdns_domain_len", "rdns_label_count",
    "rdns_tld", "rdns_first_label", "rdns_has_vpn", "rdns_has_ip_literal",
]

IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
IPTOKEN_RE = re.compile(r"\d{1,3}[-.]\d{1,3}[-.]\d{1,3}[-.]\d{1,3}")


def stream_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def ipv4_to_int(ip: str) -> int:
    return int.from_bytes(socket.inet_aton(ip), "big")


def int_to_ipv4(n: int) -> str:
    return socket.inet_ntoa(int(n).to_bytes(4, "big"))


def parse_ts(txt):
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return None


def firstval(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x


def norm_org(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:")
    # common vendor aliases seen in the audit
    aliases = {
        "watchguard": "watchguard",
        "stormshield": "stormshield",
        "netasq - secure internet connectivity": "netasq",
        "netasq": "netasq",
        "microsoft corporation": "microsoft",
        "microsoft": "microsoft",
        "digicert inc": "digicert",
        "digicert, inc": "digicert",
        "digicert, inc.": "digicert",
        "digicert": "digicert",
        "sectigo limited": "sectigo",
        "sectigo": "sectigo",
        "go daddy.com, inc": "godaddy",
        "go daddy.com, inc.": "godaddy",
        "godaddy.com, inc": "godaddy",
        "godaddy.com, inc.": "godaddy",
        "thegodaddy": "godaddy",
        "let's encrypt": "letsencrypt",
        "lets encrypt": "letsencrypt",
        "globalsign nv-sa": "globalsign",
        "thevpn": "thevpn",
    }
    return aliases.get(s, s)


def norm_server(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if s in {"", "none", "null", "-", "0.0.0.0", "xxxx", "xxxxxxxx-xxxxx"}:
        return "other"
    # product family: first token before /, space, or version number
    family = re.split(r"[/\s]", s, maxsplit=1)[0]
    family = family.strip(" .,:;")
    aliases = {
        "microsoft-httpapi": "microsoft-httpapi",
        "microsoft-iis": "microsoft-iis",
        "apache": "apache",
        "nginx": "nginx",
        "openvpn-as": "openvpn-as",
        "lighttpd": "lighttpd",
        "npppd": "npppd",
        "kestrel": "kestrel",
    }
    return aliases.get(family, family)


def norm_content_type(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).split(";")[0].strip().lower()
    return s or None


def flatten_tls(entry, source: str):
    """Return (features, fingerprint or None) for one TLS-scan row."""
    feats = dict.fromkeys(TLS_KEYS)
    d = entry.get("data") or {}
    hb = d.get("Heartbleed") if isinstance(d.get("Heartbleed"), dict) else {}
    res = hb.get("result") if isinstance(hb.get("result"), dict) else None
    hlog = (res or {}).get("handshake_log")
    feats["handshake_present"] = hlog is not None
    if not hlog:
        return feats, None

    sh = hlog.get("server_hello") or {}
    feats["tls_version"] = (sh.get("version") or {}).get("name")
    feats["tls_cipher"] = (sh.get("cipher_suite") or {}).get("name")
    feats["tls_secure_renegotiation"] = sh.get("secure_renegotiation")
    feats["tls_ticket"] = sh.get("ticket")
    feats["tls_ocsp_stapling"] = sh.get("ocsp_stapling")
    feats["tls_heartbeat_ext"] = sh.get("heartbeat")
    feats["tls_extended_master_secret"] = sh.get("extended_master_secret")

    cke = hlog.get("client_key_exchange") or {}
    rp = cke.get("rsa_params") if isinstance(cke.get("rsa_params"), dict) else None
    feats["cke_rsa_present"] = rp is not None
    feats["cke_rsa_params_len"] = rp.get("length") if rp else None

    sc = hlog.get("server_certificates") or {}
    parsed = ((sc.get("certificate") or {}).get("parsed")) or None
    feats["cert_present"] = parsed is not None
    if not parsed:
        return feats, None

    fp = parsed.get("fingerprint_sha256")
    val = parsed.get("validity") or {}
    start, end = parse_ts(val.get("start")), parse_ts(val.get("end"))
    scan_date = SCAN_DATES[source]
    if start and end:
        feats["cert_validity_days"] = (end - start).days
        feats["cert_age_days"] = (scan_date - start).days
        feats["cert_expired"] = end < scan_date

    sig = parsed.get("signature") or {}
    feats["cert_self_signed"] = sig.get("self_signed")
    feats["cert_sig_valid"] = sig.get("valid")
    feats["cert_sig_algorithm"] = (parsed.get("signature_algorithm") or {}).get("name")
    ski = parsed.get("subject_key_info") or {}
    feats["cert_key_algorithm"] = (ski.get("key_algorithm") or {}).get("name")
    feats["cert_key_bits"] = (ski.get("rsa_public_key") or {}).get("length")
    feats["cert_key_curve"] = (ski.get("ecdsa_public_key") or {}).get("curve")

    sub, iss = parsed.get("subject") or {}, parsed.get("issuer") or {}
    cn = firstval(sub.get("common_name"))
    feats["cert_subject_cn"] = cn
    feats["cert_subject_org"] = firstval(sub.get("organization"))
    feats["cert_issuer_cn"] = firstval(iss.get("common_name"))
    feats["cert_issuer_org"] = firstval(iss.get("organization"))
    feats["cert_subject_eq_issuer"] = sub == iss
    feats["cert_cn_is_ipv4"] = bool(cn and IPV4_RE.match(cn))
    feats["cert_cn_has_softether"] = bool(cn and "softether" in cn.lower())
    feats["cert_cn_localhost"] = bool(cn and cn.lower() in ("localhost", "user.local"))

    names = parsed.get("names") or []
    feats["cert_names_count"] = len(names)
    feats["cert_has_wildcard"] = any("*" in str(n) for n in names)
    exts = parsed.get("extensions") or {}
    feats["cert_san_dns_count"] = len((exts.get("subject_alt_name") or {}).get("dns_names") or [])
    feats["cert_is_ca"] = (exts.get("basic_constraints") or {}).get("is_ca")
    ku = exts.get("key_usage") or {}
    feats["cert_key_encipherment"] = ku.get("key_encipherment")
    feats["cert_digital_signature"] = ku.get("digital_signature")
    feats["cert_server_auth"] = (exts.get("extended_key_usage") or {}).get("server_auth")
    feats["cert_validation_level"] = parsed.get("validation_level")
    return feats, fp


def flatten_get(entry, prefix: str):
    out = dict.fromkeys(HTTPS_KEYS if prefix == "get_https" else HTTP_KEYS)
    probe = (entry.get("data") or {}).get("http") or {}
    out[prefix + "_probe_status"] = probe.get("status")
    res = probe.get("result") if isinstance(probe.get("result"), dict) else {}
    resp = res.get("response") if isinstance(res.get("response"), dict) else None
    out[prefix + "_response"] = resp is not None
    if not resp:
        return out
    code = resp.get("status_code")
    if prefix == "get_https":
        out["get_https_status_code"] = code
        out["get_https_status_class"] = f"{code // 100}xx" if isinstance(code, int) else None
        headers = resp.get("headers") or {}
        out["get_https_server"] = firstval(headers.get("server"))
        out["get_https_content_type"] = firstval(headers.get("content_type"))
        out["get_https_header_count"] = len(headers)
        body = resp.get("body")
        out["get_https_body_len"] = len(body) if isinstance(body, str) else None
    else:
        out["get_http_status_code"] = code
        out["get_http_server"] = firstval((resp.get("headers") or {}).get("server"))
    return out


def tls_score(feats, source):
    return (
        bool(feats.get("handshake_present")),
        bool(feats.get("cert_present")),
        sum(v is not None for v in feats.values()),
        1 if source == "openvpn" else 0,
    )


def get_score(feats, source):
    return (
        bool(feats.get("get_https_response") or feats.get("get_http_response")),
        bool(feats.get("get_https_probe_status") == "success" or feats.get("get_http_probe_status") == "success"),
        sum(v is not None for v in feats.values()),
        1 if source == "openvpn" else 0,
    )


def load_ipv4_sorted(path: Path):
    s = pd.read_csv(path, header=None, names=["ip"], dtype="string")["ip"].dropna()
    o = s.str.split(".", expand=True).astype("int64").to_numpy()
    return np.unique((o[:, 0] << 24) | (o[:, 1] << 16) | (o[:, 2] << 8) | o[:, 3])


def load_ipasn(path: Path):
    starts, ends, asns, plens = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            prefix, asn = line.rstrip().split("\t")[:2]
            ip, length = prefix.split("/")
            length = int(length)
            n = ipv4_to_int(ip)
            starts.append(n)
            ends.append(n + (1 << (32 - length)) - 1)
            asns.append(int(asn))
            plens.append(length)
    starts = np.asarray(starts, np.int64)
    ends = np.asarray(ends, np.int64)
    asns = np.asarray(asns, np.int64)
    plens = np.asarray(plens, np.int8)
    order = np.lexsort((-ends, starts))
    return starts[order], ends[order], asns[order], plens[order]


def main() -> None:
    t0 = time.time()
    base_parts = []
    for name, fname in [
        ("train", "train_70.csv"),
        ("test", "test_20.csv"),
        ("validation", "validation_10_unseen.csv"),
    ]:
        split_path = SPLIT_DIR / fname
        if not split_path.exists():
            split_path = SPLIT_DIR / (fname + ".gz")
        d = pd.read_csv(
            split_path,
            usecols=["ip", "label", "group_id", "group_cert_fp",
                     "in_openvpn_tcp", "in_sstp", "in_openvpn_udp"],
            dtype={"ip": "string", "group_id": "string", "group_cert_fp": "string"},
        )
        d["split"] = name
        base_parts.append(d)
    base = pd.concat(base_parts, ignore_index=True)
    base["ip_int"] = base["ip"].map(ipv4_to_int)
    base = base.set_index("ip_int", drop=False)
    u3_set = set(base.index)
    print(f"base rows: {len(base):,}")

    # ---- TLS best-record merge -----------------------------------------
    tls_store: dict[int, dict] = {}
    tls_source: dict[int, str] = {}
    replaced = 0
    added = 0
    for source, path in [
        ("openvpn", OUT / "ipv4/OpenVPN/ipv4-openvpn-tcp.vpn.2022-08-01-1935-tls.txt"),
        ("sstp", OUT / "ipv4/SSTP/ipv4-sstp.vpn.2022-08-09-1007-tls.txt"),
    ]:
        n = 0
        for e in stream_jsonl(path):
            ipi = ipv4_to_int(e["ip"])
            if ipi not in u3_set:
                continue
            feats, _fp = flatten_tls(e, source)
            if ipi not in tls_store:
                tls_store[ipi] = feats
                tls_source[ipi] = source
                added += 1
            else:
                old_score = tls_score(tls_store[ipi], tls_source[ipi])
                new_score = tls_score(feats, source)
                if new_score > old_score:
                    tls_store[ipi] = feats
                    tls_source[ipi] = source
                    replaced += 1
            n += 1
        print(f"TLS {source}: kept {n:,} rows")
    print(f"TLS best-record replacements: {replaced:,}")
    tls_df = pd.DataFrame.from_dict(tls_store, orient="index").reindex(base.index)
    tls_df["tls_source"] = pd.Series(tls_source).reindex(base.index)

    # ---- GET best-record + field fill ----------------------------------
    get_raw: dict[int, dict[str, dict[str, dict]]] = {}
    for source, prefix, path in [
        ("openvpn", "get_https", OUT / "ipv4/OpenVPN/ipv4-openvpn-tcp.vpn.2022-08-01-1935-https.txt"),
        ("openvpn", "get_http", OUT / "ipv4/OpenVPN/ipv4-openvpn-tcp.vpn.2022-08-01-1935-http.txt"),
        ("sstp", "get_https", OUT / "ipv4/SSTP/ipv4-sstp.vpn.2022-08-09-1007-https.txt"),
        ("sstp", "get_http", OUT / "ipv4/SSTP/ipv4-sstp.vpn.2022-08-09-1007-http.txt"),
    ]:
        n = 0
        for e in stream_jsonl(path):
            ipi = ipv4_to_int(e["ip"])
            if ipi not in u3_set:
                continue
            feats = flatten_get(e, prefix)
            get_raw.setdefault(ipi, {}).setdefault(prefix, {})[source] = feats
            n += 1
        print(f"GET {source} {prefix}: {n:,} rows scanned")

    get_rows = {}
    for ipi in base.index:
        row = {}
        for prefix, keys in [("get_https", HTTPS_KEYS), ("get_http", HTTP_KEYS)]:
            records = get_raw.get(ipi, {}).get(prefix, {})
            if not records:
                row.update(dict.fromkeys(keys))
                continue
            ordered = sorted(records.items(), key=lambda kv: get_score(kv[1], kv[0]), reverse=True)
            best_source, best = ordered[0]
            merged = dict(best)
            for _src, other in ordered[1:]:
                for k, v in other.items():
                    if merged.get(k) is None and v is not None:
                        merged[k] = v
            # keep response flag consistent with the merged status code
            if prefix == "get_https" and merged.get("get_https_status_code") is not None:
                merged["get_https_response"] = True
            if prefix == "get_http" and merged.get("get_http_status_code") is not None:
                merged["get_http_response"] = True
            row.update(merged)
        get_rows[ipi] = row
    get_df = pd.DataFrame.from_dict(get_rows, orient="index").reindex(base.index)

    def best_source(ipi, prefix):
        records = get_raw.get(ipi, {}).get(prefix, {})
        if not records:
            return None
        return sorted(records.items(), key=lambda kv: get_score(kv[1], kv[0]), reverse=True)[0][0]

    # store per-prefix source for audit only
    for prefix, col in [("get_https", "get_https_source"), ("get_http", "get_http_source")]:
        get_df[col] = pd.Series({ipi: best_source(ipi, prefix) for ipi in base.index}).reindex(base.index)

    # ---- rDNS ----------------------------------------------------------
    rdns_store = {}
    for e in stream_jsonl(OUT / "ipv4/rDNS/ipv4-all.shuf.rdns.noerror.out"):
        rev = e["name"].replace(".in-addr.arpa.", "")
        ipi = ipv4_to_int(".".join(rev.split(".")[::-1]))
        if ipi not in u3_set:
            continue
        doms = [str(a.get("data", "")).rstrip(".") for a in ((e.get("data") or {}).get("answers") or [])]
        doms = [d for d in doms if d]
        if not doms:
            continue
        d0 = doms[0]
        labels = d0.split(".")
        rdns_store[ipi] = {
            "rdns_present": True,
            "rdns_domain_count": len(doms),
            "rdns_domain_len": len(d0),
            "rdns_label_count": len(labels),
            "rdns_tld": labels[-1].lower(),
            "rdns_first_label": labels[0].lower(),
            "rdns_has_vpn": any("vpn" in d.lower() for d in doms),
            "rdns_has_ip_literal": any(bool(IPTOKEN_RE.search(d)) for d in doms),
        }
    rdns_df = pd.DataFrame.from_dict(rdns_store, orient="index").reindex(base.index)
    rdns_df["rdns_present"] = rdns_df["rdns_present"].fillna(False)
    print(f"rDNS IPs: {len(rdns_store):,}")

    # ---- ASN / prefix --------------------------------------------------
    v4_starts, v4_ends, v4_asns, v4_plens = load_ipasn(HIT / "ipasn_ipv4_20220725.2200.dat")
    # The stack sweep requires ascending IPs; base is in split order, so sort a view.
    ips = base.index.to_numpy()
    order = np.argsort(ips)
    sorted_ips = ips[order]
    asn_sorted = np.zeros(len(base), np.int64)
    plen_sorted = np.zeros(len(base), np.int8)
    ok_sorted = np.zeros(len(base), bool)
    stack, i, n = [], 0, len(v4_starts)
    for k, ip in enumerate(sorted_ips.tolist()):
        while i < n and v4_starts[i] <= ip:
            s = int(v4_starts[i])
            while stack and stack[-1][0] < s:
                stack.pop()
            stack.append((int(v4_ends[i]), int(v4_asns[i]), int(v4_plens[i])))
            i += 1
        while stack and stack[-1][0] < ip:
            stack.pop()
        if stack:
            asn_sorted[k] = stack[-1][1]
            plen_sorted[k] = stack[-1][2]
            ok_sorted[k] = True
    asn_arr = np.empty(len(base), np.int64)
    plen_arr = np.empty(len(base), np.int8)
    asn_ok = np.empty(len(base), bool)
    asn_arr[order] = asn_sorted
    plen_arr[order] = plen_sorted
    asn_ok[order] = ok_sorted
    print(f"ASN resolved: {100 * asn_ok.mean():.2f}%")

    # ---- assemble ------------------------------------------------------
    df = base.drop(columns=["ip_int"]).copy()
    df["ip_24"] = [int_to_ipv4(x & 0xFFFFFF00) for x in df.index]
    df = pd.concat([df, tls_df, get_df, rdns_df], axis=1)
    df["asn"] = asn_arr
    df["prefix_len"] = plen_arr
    df.loc[~asn_ok, "asn"] = None

    # normalized categorical variants for modeling
    df["cert_issuer_org_norm"] = df["cert_issuer_org"].map(norm_org)
    df["cert_subject_org_norm"] = df["cert_subject_org"].map(norm_org)
    df["get_https_server_family"] = df["get_https_server"].map(norm_server)
    df["get_https_content_type_norm"] = df["get_https_content_type"].map(norm_content_type)
    df["cert_validation_level_norm"] = df["cert_validation_level"].map(
        lambda v: str(v).lower() if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
    )
    for c in ["tls_version", "tls_cipher", "cert_sig_algorithm", "cert_key_algorithm", "cert_key_curve",
              "rdns_tld", "rdns_first_label"]:
        df[c] = df[c].map(lambda v: str(v).lower() if v is not None and not (isinstance(v, float) and np.isnan(v)) else None)

    # ---- schema --------------------------------------------------------
    forbidden = [
        "ip", "ip_24", "label", "group_cert_fp", "group_id", "split",
        "in_openvpn_tcp", "in_sstp", "in_openvpn_udp",
        "tls_source", "get_https_source", "get_http_source",
        "cert_subject_cn", "cert_issuer_cn", "asn",
    ]
    drop_redundant = [
        "tls_version", "tls_secure_renegotiation", "tls_ticket", "tls_extended_master_secret",
        "cert_present", "cert_key_algorithm", "cert_cn_localhost",
        "cert_key_encipherment", "cert_digital_signature", "cert_server_auth",
        "cert_key_curve", "cke_rsa_params_len", "cert_san_dns_count", "cert_names_count",
        "cert_has_wildcard", "cert_expired", "cert_is_ca", "cert_sig_valid",
        "cert_sig_algorithm", "cert_key_bits", "cke_rsa_present", "tls_heartbeat_ext",
        "get_https_body_len", "get_https_status_code", "get_http_probe_status",
        "get_http_response", "get_http_status_code", "get_http_server",
        "rdns_present", "rdns_domain_count", "rdns_domain_len", "rdns_label_count",
        "rdns_tld", "rdns_first_label", "rdns_has_vpn", "rdns_has_ip_literal",
        "prefix_len",
    ]
    all_cols = [c for c in df.columns if c not in forbidden]
    allowed = [c for c in all_cols if c not in drop_redundant]
    schema = {
        "universe": "U3 = H(openvpn_tcp) union H(sstp), IPv4",
        "row_count": int(len(df)),
        "merge_rules": {
            "tls": "best record by (handshake, cert, non-null count, openvpn preference)",
            "get": "best record by (response, success, non-null count, openvpn preference), then field-fill",
        },
        "forbidden_columns": forbidden,
        "forbidden_reasons": {
            "in_openvpn_tcp": "label component",
            "in_sstp": "label component",
            "in_openvpn_udp": "semi-leaky same-protocol OpenVPN UDP probe",
            "group_id": "grouping key",
            "group_cert_fp": "grouping key",
            "ip": "identifier",
            "ip_24": "identifier",
            "tls_source": "source provenance; perfectly label-correlated",
            "get_https_source": "source provenance",
            "get_http_source": "source provenance",
            "cert_subject_cn": "raw high-cardinality string",
            "cert_issuer_cn": "raw high-cardinality string",
            "asn": "high-cardinality network identifier; dropped from clean model",
        },
        "dropped_redundant": drop_redundant,
        "allowed_features": allowed,
        "categorical_features": [
            c for c in allowed if c in {
                "tls_cipher", "cert_validation_level", "cert_validation_level_norm",
                "cert_issuer_org", "cert_issuer_org_norm",
                "cert_subject_org", "cert_subject_org_norm",
                "get_https_server", "get_https_server_family",
                "get_https_content_type", "get_https_content_type_norm",
                "get_https_status_class", "get_https_probe_status",
            }
        ],
        "notes": [
            "cert_age_days is computed relative to the scan date, not a fixed reference date",
            "prefix_len, rDNS, and ASN are retained in features_v2.csv for audit but dropped from the compact model",
            "tls_source/get_https_source/get_http_source are audit-only",
        ],
    }
    (V2 / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    # ---- write outputs -------------------------------------------------
    key_cols = ["ip", "ip_24", "label", "group_id", "group_cert_fp", "split",
                "in_openvpn_tcp", "in_sstp", "in_openvpn_udp"]
    ordered = key_cols + [c for c in df.columns if c not in key_cols]
    df = df[ordered]
    df.to_csv(V2 / "features_v2.csv", index=False)
    model_cols = ["ip", "label", "group_id", "split"] + [c for c in allowed if c != "split"]
    df[model_cols].to_csv(V2 / "features_model.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "tls_best_record_replacements": int(replaced),
        "tls_source_counts": df["tls_source"].value_counts(dropna=False).to_dict(),
        "get_https_source_counts": df["get_https_source"].value_counts(dropna=False).to_dict(),
        "get_http_source_counts": df["get_http_source"].value_counts(dropna=False).to_dict(),
        "coverage": {
            "handshake_present_pct": round(100 * df["handshake_present"].fillna(False).mean(), 2),
            "cert_present_pct": round(100 * df["cert_present"].fillna(False).mean(), 2),
            "get_https_response_pct": round(100 * df["get_https_response"].fillna(False).mean(), 2),
            "rdns_present_pct": round(100 * df["rdns_present"].fillna(False).mean(), 2),
            "asn_present_pct": round(100 * df["asn"].notna().mean(), 2),
        },
        "allowed_features": allowed,
        "forbidden_columns": forbidden,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (V2 / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", V2)


if __name__ == "__main__":
    main()
